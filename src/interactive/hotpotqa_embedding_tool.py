"""HotpotQA embedding retrieval for the shared ToolRegistry.

This is a thin adapter around the public-context embedding index.  It keeps
FlowSteer's ToolRegistry/ReAct boundary and SkillFlow's dynamic search/read
execution shape: retrieval happens only when an Agent dispatches a Tool
action, never as evaluation-time prefetch.

The registry supports the existing task-scoped public-passage index and the
global paired QA-memory indices.  In the passage condition, ``task_id``
selects the current task's public context.  In the QA-memory condition it is
only evaluation-call provenance in the receipt and is never passed to global
``search(query, k)`` or ``read(memory_id)``.  Results are projected through an
explicit corpus-specific allowlist.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import inspect
import json
import math
import re
from typing import Protocol
import unicodedata

from .agent_runtime import AgentRequest
from .hotpotqa_qa_memory_index import QA_MEMORY_CORPUS_VERSION
from .hotpotqa_transductive_qa_memory_index import (
    TRANSDUCTIVE_EVALUATION_REGIME,
    TRANSDUCTIVE_QA_MEMORY_CORPUS_VERSION,
)
from .react_execution import ToolReactExecutionAdapter
from .tool_runtime import (
    ToolCapability,
    ToolRegistration,
    ToolRegistry,
    ToolRequest,
    ToolResult,
)


HOTPOTQA_RETRIEVAL_TOOL_ID = "qa-retrieval"
HOTPOTQA_QA_MEMORY_TOOL_ID = "hotpotqa.qa_memory"
HOTPOTQA_DATASET_SCOPE = ("hotpotqa",)

_PASSAGE_INDEX_KIND = "public_passage"
_QA_MEMORY_INDEX_KIND = "train_qa_memory"
_TRANSDUCTIVE_QA_MEMORY_INDEX_KIND = "transductive_qa_memory"
_QA_MEMORY_RETRIEVAL_SUFFICIENCY = frozenset({"supported", "unsupported"})


def _is_qa_memory_index_kind(index_kind: str) -> bool:
    return index_kind in {
        _QA_MEMORY_INDEX_KIND,
        _TRANSDUCTIVE_QA_MEMORY_INDEX_KIND,
    }


def _normalized_retrieval_query(query: str) -> str:
    """Canonicalize only for duplicate-request admission, not retrieval.

    DIRECT_REUSE: ``qa_tool_adapter._normalized_retrieval_query`` in the
    current FlowSteer QA Tool path.  The original query remains unchanged in
    the request and receipt.
    """

    normalized = unicodedata.normalize("NFKC", query).casefold()
    return " ".join(re.findall(r"\w+", normalized, flags=re.UNICODE))


@dataclass(frozen=True, slots=True)
class _HotpotRetrievalState:
    """Public bounded search/read state adapted from FlowSteer's QA adapter."""

    search_queries: tuple[str, ...]
    latest_search_doc_ids: tuple[str, ...]
    read_doc_ids: tuple[str, ...]
    dispatched_tool_calls: int
    latest_successful_operation: str | None

    @property
    def latest_unread_doc_ids(self) -> tuple[str, ...]:
        read = frozenset(self.read_doc_ids)
        return tuple(
            doc_id for doc_id in self.latest_search_doc_ids if doc_id not in read
        )


def _public_retrieval_state(
    observations: Sequence[Mapping[str, object]],
    *,
    tool_id: str = HOTPOTQA_RETRIEVAL_TOOL_ID,
) -> _HotpotRetrievalState:
    search_queries: list[str] = []
    latest_search_doc_ids: list[str] = []
    read_doc_ids: list[str] = []
    dispatched_tool_calls = 0
    latest_successful_operation: str | None = None
    for observation in observations:
        if (
            observation.get("tool_id") == tool_id
            and observation.get("observation_status") in {"success", "tool_error"}
        ):
            dispatched_tool_calls += 1
        if observation.get("observation_status") != "success":
            continue
        result = observation.get("result")
        if not isinstance(result, Mapping):
            continue
        if result.get("operation") == "search":
            latest_successful_operation = "search"
            latest_search_doc_ids = []
            query = result.get("query")
            if isinstance(query, str) and query.strip():
                search_queries.append(query.strip())
            raw_doc_ids = result.get("memory_ids", result.get("doc_ids", ()))
            if isinstance(raw_doc_ids, (list, tuple)):
                latest_search_doc_ids.extend(
                    str(item)
                    for item in raw_doc_ids
                    if isinstance(item, str) and item.strip()
                )
        elif result.get("operation") == "read":
            latest_successful_operation = "read"
            doc_id = result.get("memory_id", result.get("doc_id"))
            if (
                isinstance(doc_id, str)
                and doc_id.strip()
                and doc_id.strip() not in read_doc_ids
            ):
                read_doc_ids.append(doc_id.strip())
    return _HotpotRetrievalState(
        search_queries=tuple(search_queries),
        latest_search_doc_ids=tuple(dict.fromkeys(latest_search_doc_ids)),
        read_doc_ids=tuple(read_doc_ids),
        dispatched_tool_calls=dispatched_tool_calls,
        latest_successful_operation=latest_successful_operation,
    )


class _EmbeddingIndex(Protocol):
    """Common manifest boundary for the two supported dense indices."""

    @property
    def manifest(self) -> object:
        ...

    def search(self, *args: object) -> Sequence[object]:
        ...

    def read(self, *args: object) -> object:
        ...


def _field(value: object, name: str) -> object:
    if isinstance(value, Mapping):
        if name not in value:
            raise ValueError(f"retrieval result is missing public field {name!r}")
        return value[name]
    if not hasattr(value, name):
        raise ValueError(f"retrieval result is missing public field {name!r}")
    return getattr(value, name)


def _optional_manifest_field(manifest: object, *names: str) -> object | None:
    for name in names:
        if isinstance(manifest, Mapping) and name in manifest:
            return manifest[name]
        if hasattr(manifest, name):
            return getattr(manifest, name)
    return None


def _required_text(value: object, *, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be non-empty text")
    return value.strip()


def _positive_integer(value: object, *, field_name: str) -> int:
    if type(value) is not int or value < 1:
        raise ValueError(f"{field_name} must be a positive integer")
    return value


def _index_identity(
    index: _EmbeddingIndex,
    *,
    index_kind: str,
) -> dict[str, object]:
    """Return only public, replay-relevant manifest fields."""

    manifest = index.manifest
    index_id = _required_text(
        _optional_manifest_field(manifest, "index_id"),
        field_name="retrieval manifest index_id",
    )
    identity: dict[str, object] = {
        "index_id": index_id,
        "corpus_kind": index_kind,
    }
    aliases = {
        "schema_version": ("schema_version",),
        "corpus_version": ("corpus_version",),
        "source": ("source", "source_dataset"),
        "split": ("split", "source_split"),
        "embedding_model": ("embedding_model", "embedding_model_id"),
        "embedding_dimension": ("embedding_dimension", "dimension"),
        "normalized": ("normalized", "embedding_normalized"),
        "similarity": ("similarity", "similarity_metric"),
        "document_count": ("document_count", "doc_count"),
        "passage_count": ("passage_count",),
        "train_record_count": ("train_record_count",),
        "source_record_count": ("source_record_count",),
        "source_train_count": ("source_train_count",),
        "source_evaluation_count": ("source_evaluation_count",),
        "source_splits": ("source_splits",),
        "unique_source_count": ("unique_source_count",),
        "cycled_record_count": ("cycled_record_count",),
        "paraphrase_count": ("paraphrase_count",),
        "heldout_validation_count": ("heldout_validation_count",),
        "validation_overlap_count": ("validation_overlap_count",),
        "frozen_validation_count": ("frozen_validation_count",),
        "evaluation_overlap_count": ("evaluation_overlap_count",),
        "contains_evaluation_answers": ("contains_evaluation_answers",),
        "evaluation_regime": ("evaluation_regime",),
        "official_heldout_eligible": ("official_heldout_eligible",),
        "paraphrase_versions": ("paraphrase_versions",),
        "paraphrase_provenances": ("paraphrase_provenances",),
        "frozen_top_k": ("frozen_top_k",),
    }
    for public_name, candidate_names in aliases.items():
        raw = _optional_manifest_field(manifest, *candidate_names)
        if raw is not None:
            identity[public_name] = list(raw) if isinstance(raw, tuple) else raw
    return identity


def _call_parameter_names(callback: object, *, name: str) -> tuple[str, ...]:
    if not callable(callback):
        raise TypeError(f"embedding index {name} must be callable")
    try:
        signature = inspect.signature(callback)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"embedding index {name} signature is unavailable") from exc
    parameters: list[str] = []
    for parameter in signature.parameters.values():
        if parameter.kind in {
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        }:
            raise TypeError(f"embedding index {name} cannot use variadic arguments")
        parameters.append(parameter.name)
    return tuple(parameters)


def _index_kind(index: _EmbeddingIndex) -> str:
    """Fail closed unless manifest kind and callable signatures agree."""

    manifest = index.manifest
    corpus_version = _required_text(
        _optional_manifest_field(manifest, "corpus_version"),
        field_name="retrieval manifest corpus_version",
    )
    search_parameters = _call_parameter_names(index.search, name="search")
    read_parameters = _call_parameter_names(index.read, name="read")
    if corpus_version == QA_MEMORY_CORPUS_VERSION:
        if _optional_manifest_field(manifest, "source_split", "split") != "train":
            raise ValueError("QA-memory retrieval index must use the train split")
        if _optional_manifest_field(manifest, "validation_overlap_count") != 0:
            raise ValueError("QA-memory retrieval index overlaps held-out validation")
        if search_parameters != ("query", "k") or read_parameters != ("memory_id",):
            raise TypeError("QA-memory manifest and global search/read signatures differ")
        return _QA_MEMORY_INDEX_KIND
    if corpus_version == TRANSDUCTIVE_QA_MEMORY_CORPUS_VERSION:
        source_splits = _optional_manifest_field(manifest, "source_splits")
        if tuple(source_splits) != ("train", "frozen_validation"):
            raise ValueError("transductive QA-memory source splits differ")
        if _optional_manifest_field(manifest, "contains_evaluation_answers") is not True:
            raise ValueError("transductive QA-memory must declare evaluation answers")
        if (
            _optional_manifest_field(manifest, "evaluation_regime")
            != TRANSDUCTIVE_EVALUATION_REGIME
        ):
            raise ValueError("transductive QA-memory evaluation regime differs")
        if _optional_manifest_field(manifest, "official_heldout_eligible") is not False:
            raise ValueError("transductive QA-memory cannot be held-out eligible")
        frozen_validation_count = _optional_manifest_field(
            manifest, "frozen_validation_count"
        )
        if (
            not isinstance(frozen_validation_count, int)
            or frozen_validation_count < 1
            or _optional_manifest_field(manifest, "evaluation_overlap_count")
            != frozen_validation_count
        ):
            raise ValueError("transductive QA-memory evaluation overlap differs")
        if search_parameters != ("query", "k") or read_parameters != ("memory_id",):
            raise TypeError("QA-memory manifest and global search/read signatures differ")
        return _TRANSDUCTIVE_QA_MEMORY_INDEX_KIND
    if search_parameters != ("task_id", "query", "k") or read_parameters not in {
        ("task_id", "doc_id"),
        ("task_id", "passage_id"),
    }:
        raise TypeError("passage manifest and task-scoped search/read signatures differ")
    return _PASSAGE_INDEX_KIND


def _validate_action(request: ToolRequest, expected_action: str) -> None:
    if request.action != expected_action:
        raise ValueError(
            f"{expected_action} backend received incompatible action "
            f"{request.action!r}"
        )


def _public_hit(raw_hit: object, *, expected_rank: int) -> dict[str, object]:
    passage_id = _required_text(
        _field(raw_hit, "passage_id"), field_name="search hit passage_id"
    )
    document_id = _required_text(
        _field(raw_hit, "document_id"), field_name="search hit document_id"
    )
    title = _required_text(_field(raw_hit, "title"), field_name="search hit title")
    snippet = _required_text(
        _field(raw_hit, "snippet"), field_name="search hit snippet"
    )
    rank = _field(raw_hit, "rank")
    if type(rank) is not int or rank != expected_rank:
        raise ValueError("embedding search ranks must be contiguous and one-based")
    similarity = _field(raw_hit, "similarity")
    if type(similarity) not in {int, float} or not math.isfinite(float(similarity)):
        raise ValueError("embedding search similarity must be finite")
    return {
        # ``doc_id`` is the canonical read handle exposed to the Agent.  The
        # underlying index calls this passage_id because one document may be
        # split into multiple independently readable passages.
        "doc_id": passage_id,
        "document_id": document_id,
        "title": title,
        "snippet": snippet,
        "similarity": float(similarity),
        "rank": rank,
    }


def _public_memory_hit(
    raw_hit: object,
    *,
    expected_rank: int,
) -> dict[str, object]:
    memory_id = _required_text(
        _field(raw_hit, "memory_id"), field_name="search hit memory_id"
    )
    rank = _field(raw_hit, "rank")
    if type(rank) is not int or rank != expected_rank:
        raise ValueError("QA-memory search ranks must be contiguous and one-based")
    similarity = _field(raw_hit, "similarity")
    if type(similarity) not in {int, float} or not math.isfinite(float(similarity)):
        raise ValueError("QA-memory search similarity must be finite")
    return {
        "memory_id": memory_id,
        "source_train_task_id": _required_text(
            _field(raw_hit, "source_train_task_id"),
            field_name="search hit source_train_task_id",
        ),
        "paraphrase_question": _required_text(
            _field(raw_hit, "paraphrase_question"),
            field_name="search hit paraphrase_question",
        ),
        "paraphrase_answer_statement": _required_text(
            _field(raw_hit, "paraphrase_answer_statement"),
            field_name="search hit paraphrase_answer_statement",
        ),
        "similarity": float(similarity),
        "rank": rank,
    }


def _public_memory(raw_memory: object, *, expected_memory_id: str) -> dict[str, object]:
    memory_id = _required_text(
        _field(raw_memory, "memory_id"), field_name="read memory memory_id"
    )
    if memory_id != expected_memory_id:
        raise ValueError("QA-memory read returned a different memory_id")
    cycled = _field(raw_memory, "cycled")
    if not isinstance(cycled, bool):
        raise TypeError("read memory cycled must be a boolean")
    return {
        "memory_id": memory_id,
        "source_train_task_id": _required_text(
            _field(raw_memory, "source_train_task_id"),
            field_name="read memory source_train_task_id",
        ),
        "base_task_id": _required_text(
            _field(raw_memory, "base_task_id"),
            field_name="read memory base_task_id",
        ),
        "cycled": cycled,
        "paraphrase_question": _required_text(
            _field(raw_memory, "paraphrase_question"),
            field_name="read memory paraphrase_question",
        ),
        "paraphrase_answer_statement": _required_text(
            _field(raw_memory, "paraphrase_answer_statement"),
            field_name="read memory paraphrase_answer_statement",
        ),
        "canonical_answer": _required_text(
            _field(raw_memory, "canonical_answer"),
            field_name="read memory canonical_answer",
        ),
        "paraphrase_version": _required_text(
            _field(raw_memory, "paraphrase_version"),
            field_name="read memory paraphrase_version",
        ),
        "paraphrase_provenance": _required_text(
            _field(raw_memory, "paraphrase_provenance"),
            field_name="read memory paraphrase_provenance",
        ),
    }


@dataclass(slots=True)
class HotpotQAEmbeddingToolBackend:
    """Dispatch either task-scoped passage or global train-memory retrieval."""

    index: _EmbeddingIndex
    task_id: str
    frozen_top_k: int
    index_identity: Mapping[str, object]
    index_kind: str
    _search_returned_doc_ids: set[str] = field(
        default_factory=set,
        init=False,
        repr=False,
    )

    async def invoke(self, request: ToolRequest) -> ToolResult:
        if request.action == "search":
            return await self._search(request)
        if request.action == "read":
            return await self._read(request)
        raise ValueError(
            f"retrieval backend received unsupported action {request.action!r}"
        )

    async def _search(self, request: ToolRequest) -> ToolResult:
        _validate_action(request, "search")
        if set(request.arguments) != {"query", "k"}:
            raise ValueError("search arguments must contain exactly query and k")
        query = request.arguments["query"]
        k = request.arguments["k"]
        if not isinstance(query, str) or not query.strip():
            raise ValueError("search query must be non-empty text")
        if type(k) is not int or k != self.frozen_top_k:
            raise ValueError("search k differs from the frozen embedding top-k")

        if _is_qa_memory_index_kind(self.index_kind):
            candidate = self.index.search(query, k)
        else:
            candidate = self.index.search(self.task_id, query, k)
        raw_hits = await candidate if inspect.isawaitable(candidate) else candidate
        if not isinstance(raw_hits, Sequence) or isinstance(raw_hits, (str, bytes)):
            raise TypeError("embedding search must return a sequence of hits")
        if len(raw_hits) > self.frozen_top_k:
            raise ValueError("embedding search returned more than frozen top-k hits")
        if _is_qa_memory_index_kind(self.index_kind):
            hits = [
                _public_memory_hit(raw_hit, expected_rank=rank)
                for rank, raw_hit in enumerate(raw_hits, start=1)
            ]
            resource_id_key = "memory_ids"
            resource_ids = [str(hit["memory_id"]) for hit in hits]
        else:
            hits = [
                _public_hit(raw_hit, expected_rank=rank)
                for rank, raw_hit in enumerate(raw_hits, start=1)
            ]
            resource_id_key = "doc_ids"
            resource_ids = [str(hit["doc_id"]) for hit in hits]
        if len(set(resource_ids)) != len(resource_ids):
            raise ValueError("embedding search returned duplicate resource IDs")
        self._search_returned_doc_ids.update(resource_ids)
        return ToolResult(
            {
                "operation": "search",
                "task_id": self.task_id,
                "retrieval_index": dict(self.index_identity),
                "query": query,
                "k": k,
                resource_id_key: resource_ids,
                "hits": hits,
            }
        )


    async def _read(self, request: ToolRequest) -> ToolResult:
        _validate_action(request, "read")
        id_field = (
            "memory_id"
            if _is_qa_memory_index_kind(self.index_kind)
            else "doc_id"
        )
        if set(request.arguments) != {id_field}:
            raise ValueError(f"read arguments must contain exactly {id_field}")
        resource_id = _required_text(
            request.arguments[id_field], field_name=f"read {id_field}"
        )
        if resource_id not in self._search_returned_doc_ids:
            raise ValueError(
                f"read {id_field} was not returned by a successful search"
            )

        if _is_qa_memory_index_kind(self.index_kind):
            candidate = self.index.read(resource_id)
        else:
            candidate = self.index.read(self.task_id, resource_id)
        raw_value = await candidate if inspect.isawaitable(candidate) else candidate
        if _is_qa_memory_index_kind(self.index_kind):
            public_memory = _public_memory(
                raw_value,
                expected_memory_id=resource_id,
            )
            return ToolResult(
                {
                    "operation": "read",
                    "task_id": self.task_id,
                    "retrieval_index": dict(self.index_identity),
                    "memory_id": resource_id,
                    "memory": public_memory,
                }
            )
        returned_passage_id = _required_text(
            _field(raw_value, "passage_id"),
            field_name="read passage passage_id",
        )
        if returned_passage_id != resource_id:
            raise ValueError("embedding read returned a different passage_id")
        public_passage = {
            "doc_id": returned_passage_id,
            "document_id": _required_text(
                _field(raw_value, "document_id"),
                field_name="read passage document_id",
            ),
            "title": _required_text(
                _field(raw_value, "title"), field_name="read passage title"
            ),
            "content": _required_text(
                _field(raw_value, "text"), field_name="read passage text"
            ),
        }
        return ToolResult(
            {
                "operation": "read",
                "task_id": self.task_id,
                "retrieval_index": dict(self.index_identity),
                "doc_id": resource_id,
                "passage": public_passage,
            }
        )


class HotpotQAEmbeddingReactExecutionAdapter(ToolReactExecutionAdapter):
    """Admit completion only after measured dynamic search and read.

    SkillFlow's QA retrieval environment requires Tool use before an answer.
    This narrow adapter keeps that execution boundary while leaving Agent
    roles, relations, and overall topology entirely under the Director's
    progressive Canvas search space.
    """

    def __init__(
        self,
        *,
        retrieval_query_scope: str | None = None,
        **kwargs: object,
    ) -> None:
        self._retrieval_query_scope = (
            None
            if retrieval_query_scope is None
            else _required_text(
                retrieval_query_scope,
                field_name="retrieval_query_scope",
            )
        )
        super().__init__(**kwargs)

    def _active_retrieval_tool_id(self) -> str:
        resource_ids = self._tool_registry.resource_ids
        if len(resource_ids) != 1:
            raise ValueError("HotpotQA retrieval adapter requires one Tool resource")
        return resource_ids[0]

    def _read_identifier_name(self) -> str:
        capability = self._tool_registry.require_capability(
            self._active_retrieval_tool_id()
        )
        schema = capability.action_schemas.get("read", {})
        required = schema.get("required", ()) if isinstance(schema, Mapping) else ()
        return "memory_id" if tuple(required) == ("memory_id",) else "doc_id"

    def _completion_arguments_schema(
        self,
        request: AgentRequest,
    ) -> Mapping[str, object]:
        """Bind QA-memory completion to a worker retrieval assessment.

        DIRECT_REUSE: SkillFlow's bounded Agent completion remains one
        ``StructuredAction``.  This HotpotQA compatibility schema replaces
        the unconstrained text artifact only for the QA-memory resource: the
        worker reads the complete frozen top-k group, then reports whether a
        selected record supports the current public task or the whole group
        is unsupported.  The judgment does not change AgentGraph roles,
        relations, or terminal semantics.
        """

        if self._active_retrieval_tool_id() != HOTPOTQA_QA_MEMORY_TOOL_ID:
            return super()._completion_arguments_schema(request)
        del request
        return {
            "type": "object",
            "required": ["value"],
            "additionalProperties": False,
            "properties": {
                "value": {
                    "type": "object",
                    "required": ["retrieval_sufficiency", "selected_memory_id"],
                    "additionalProperties": False,
                    "properties": {
                        "selected_memory_id": {
                            "type": ["string", "null"],
                            "description": (
                                "For supported retrieval, one exact memory_id from "
                                "the current fully read top-k group; null for an "
                                "unsupported group."
                            ),
                        },
                        "retrieval_sufficiency": {
                            "type": "string",
                            "enum": sorted(_QA_MEMORY_RETRIEVAL_SUFFICIENCY),
                            "description": (
                                "supported only when entity binding, relation, "
                                "qualifiers, and answer slot align with the "
                                "current public task; otherwise unsupported."
                            ),
                        },
                    },
                }
            },
        }

    def _allowed_tools_for_turn(
        self,
        request: AgentRequest,
        observations: list[Mapping[str, object]],
    ) -> tuple[str, ...]:
        actions, _ = self._state_conditioned_action_domain(
            request,
            observations,
        )
        if actions:
            return (self._active_retrieval_tool_id(),)
        return ()

    def _state_conditioned_action_domain(
        self,
        request: AgentRequest,
        observations: list[Mapping[str, object]],
    ) -> tuple[frozenset[tuple[str, str]], bool]:
        """Expose FlowSteer's bounded HotpotQA multi-hop action domain.

        THIN_ADAPTATION: the state transitions are ported from
        ``QARetrievalReactExecutionAdapter`` in the current FlowSteer QA path.
        This task-scoped adapter keeps the local ``doc_id``/``k`` wire and has
        no dependency on Agent role, relation, output identity, or topology.
        """

        del request
        tool_id = self._active_retrieval_tool_id()
        state = _public_retrieval_state(observations, tool_id=tool_id)
        remaining_tool_calls = max(
            0,
            self._max_tool_calls - state.dispatched_tool_calls,
        )
        successful_read_count = len(state.read_doc_ids)
        if tool_id == HOTPOTQA_QA_MEMORY_TOOL_ID:
            # NECESSARY_ADAPTATION: every member of the frozen embedding top-k
            # is a paired QA record. Completion is admitted only after
            # the worker has inspected the complete returned group, so an
            # unsupported assessment cannot be made from rank 1 alone.
            if state.latest_search_doc_ids and state.latest_unread_doc_ids:
                if remaining_tool_calls == 0:
                    return frozenset(), False
                return frozenset({(tool_id, "read")}), False
            if state.latest_search_doc_ids and not state.latest_unread_doc_ids:
                return frozenset(), True
            if remaining_tool_calls == 0:
                return frozenset(), False
            return frozenset({(tool_id, "search")}), False
        # DIRECT_REUSE: SkillFlow/FlowSteer's shared QA retrieval adapter
        # admits completion after one successful public read.  QA-memory hits
        # are complete paired-QA demonstrations rather than task-scoped
        # passage hops, so forcing a second search/read only adds an unrelated
        # example and consumes the bounded ReAct turn budget.
        if successful_read_count >= 1:
            return frozenset(), True
        if remaining_tool_calls == 0:
            return frozenset(), successful_read_count > 0
        if (
            state.latest_successful_operation == "search"
            and state.latest_unread_doc_ids
        ):
            return frozenset(
                {(tool_id, "read")}
            ), False
        return frozenset(
            {(tool_id, "search")}
        ), False

    def _contract(
        self,
        request: AgentRequest,
        observations: list[Mapping[str, object]],
    ) -> str:
        base = super()._contract(request, observations)
        tool_id = self._active_retrieval_tool_id()
        capability = self._tool_registry.require_capability(tool_id)
        schemas = capability.input_schema.get("oneOf", ())
        search_schema = next(
            (
                schema
                for schema in schemas
                if isinstance(schema, Mapping) and schema.get("title") == "search"
            ),
            {},
        )
        properties = (
            search_schema.get("properties", {})
            if isinstance(search_schema, Mapping)
            else {}
        )
        k_schema = properties.get("k", {}) if isinstance(properties, Mapping) else {}
        frozen_k = k_schema.get("const") if isinstance(k_schema, Mapping) else None
        state = _public_retrieval_state(observations, tool_id=tool_id)
        read_identifier = self._read_identifier_name()
        actions, completion_admitted = self._state_conditioned_action_domain(
            request,
            observations,
        )
        action_names = frozenset(action_name for _, action_name in actions)
        if action_names == {"read"}:
            next_action = (
                "Current action mask: read only; search and complete are not "
                f"admitted. Use exactly one unread {read_identifier} from the latest "
                "successful search: "
                + json.dumps(
                    list(state.latest_unread_doc_ids),
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
                + "."
            )
        elif action_names == {"search"} and state.read_doc_ids:
            next_action = (
                "Current action mask: search only; read and complete are not "
                "admitted. Form a distinct focused query for the missing "
                "entity, relation, or hop. A missing mention in one passage "
                "is not evidence that another named entity lacks the requested "
                "fact. Prior successful queries: "
                + json.dumps(
                    list(state.search_queries),
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
                + "."
            )
        elif completion_admitted and not actions:
            if tool_id == HOTPOTQA_QA_MEMORY_TOOL_ID:
                next_action = (
                    "Current action mask: complete only. Judge the fully read top-k "
                    "group against the current query and public task passages. "
                    "Check entity identity, predicate or relation, qualifiers, and "
                    "the requested answer slot. Use only those public inputs and the "
                    "Tool observations; do not use a reference answer or evaluator "
                    "metadata. Return retrieval_sufficiency=supported only when all "
                    "four fields align. Otherwise return unsupported so downstream "
                    "execution falls back to answering the public task directly."
                )
            else:
                next_action = (
                    "Current action mask: complete only. Return the best task "
                    "artifact permitted by the assigned contract."
                )
        else:
            next_action = (
                "Current action mask: search only; read and complete are not "
                "admitted. Supply all required search arguments."
            )
        query_scope = self._retrieval_query_scope or request.problem
        completion_example = '{"value":"evidence-supported artifact"}'
        if tool_id == HOTPOTQA_QA_MEMORY_TOOL_ID:
            completion_example = json.dumps(
                {
                    "value": {
                        "retrieval_sufficiency": "unsupported",
                        "selected_memory_id": None,
                    }
                },
                ensure_ascii=False,
                separators=(",", ":"),
            )
        return (
            base
            + "\nEmbedding retrieval query scope: "
            + json.dumps(query_scope, ensure_ascii=False)
            + ". Form search queries only from this question scope; the public "
            "passages remain task evidence and are not a static retrieval payload. "
            + "\nHotpotQA retrieval protocol: search arguments must be exactly "
            + json.dumps(
                {"query": "focused evidence query", "k": frozen_k},
                sort_keys=True,
                separators=(",", ":"),
            )
            + ". After a successful search, choose an exact returned "
            + read_identifier
            + " and call read with exactly "
            + json.dumps(
                {read_identifier: f"returned {read_identifier}"},
                sort_keys=True,
                separators=(",", ":"),
            )
            + ". Do not "
            "repeat a normalized successful query. Embedding similarity identifies "
            "semantic-neighbor QA records; it is not factual entailment for "
            "the current question. Complete only on the admitted terminal state, "
            "after the semantic-alignment check, with exactly "
            + completion_example
            + ". "
            + next_action
        )

    def _tool_action_error(
        self,
        *,
        request: AgentRequest,
        action: object,
        observations: list[Mapping[str, object]],
    ) -> str | None:
        del request
        state = _public_retrieval_state(
            observations,
            tool_id=self._active_retrieval_tool_id(),
        )
        name = getattr(action, "name", None)
        arguments = getattr(action, "arguments", None)
        if name == "search" and isinstance(arguments, Mapping):
            query = arguments.get("query")
            if isinstance(query, str):
                normalized = _normalized_retrieval_query(query)
                prior = {
                    _normalized_retrieval_query(value)
                    for value in state.search_queries
                }
                if normalized and normalized in prior:
                    return "hotpotqa_duplicate_normalized_query"
        if name == "read" and isinstance(arguments, Mapping):
            resource_id = arguments.get(self._read_identifier_name())
            if resource_id not in state.latest_unread_doc_ids:
                return "hotpotqa_read_not_latest_unread_candidate"
        return None

    def _completion_error(
        self,
        *,
        action: object,
        artifact: str,
        tool_receipts: list[dict[str, object]],
    ) -> str | None:
        del artifact
        tool_id = (
            self._active_retrieval_tool_id()
            if hasattr(self, "_tool_registry")
            else HOTPOTQA_RETRIEVAL_TOOL_ID
        )
        successful_actions = {
            str(request["action"])
            for receipt in tool_receipts
            if receipt.get("tool_id") == tool_id
            and receipt.get("error_type") is None
            and isinstance(receipt.get("result"), Mapping)
            and isinstance((request := receipt.get("request")), Mapping)
            and request.get("action") in {"search", "read"}
        }
        if "search" not in successful_actions:
            return "hotpotqa_dynamic_search_required"
        if "read" not in successful_actions:
            return "hotpotqa_dynamic_read_required"
        if tool_id == HOTPOTQA_QA_MEMORY_TOOL_ID:
            arguments = getattr(action, "arguments", None)
            value = (
                arguments.get("value")
                if isinstance(arguments, Mapping)
                else None
            )
            if not isinstance(value, Mapping) or set(value) != {
                "retrieval_sufficiency",
                "selected_memory_id",
            }:
                return "hotpotqa_qa_memory_completion_schema_invalid"
            selected_memory_id = value.get("selected_memory_id")
            sufficiency = value.get("retrieval_sufficiency")
            if not isinstance(sufficiency, str) or (
                sufficiency not in _QA_MEMORY_RETRIEVAL_SUFFICIENCY
            ):
                return "hotpotqa_qa_memory_retrieval_sufficiency_invalid"
            if sufficiency == "supported" and (
                not isinstance(selected_memory_id, str)
                or not selected_memory_id.strip()
            ):
                return "hotpotqa_qa_memory_selected_memory_id_invalid"
            if sufficiency == "unsupported" and selected_memory_id is not None:
                return "hotpotqa_qa_memory_unsupported_selection_must_be_null"
            latest_search_memory_ids: set[str] = set()
            latest_read_memory_ids: set[str] = set()
            for receipt in tool_receipts:
                if (
                    receipt.get("tool_id") != tool_id
                    or receipt.get("error_type") is not None
                ):
                    continue
                request_value = receipt.get("request")
                result_value = receipt.get("result")
                if not isinstance(request_value, Mapping) or not isinstance(
                    result_value, Mapping
                ):
                    continue
                result = result_value.get("value")
                if result_value.get("completed") is not True or not isinstance(
                    result, Mapping
                ):
                    continue
                operation = request_value.get("action")
                if operation == "search" and result.get("operation") == "search":
                    memory_ids = result.get("memory_ids")
                    hits = result.get("hits")
                    if isinstance(memory_ids, list) and isinstance(hits, list):
                        returned = {
                            item
                            for item in memory_ids
                            if isinstance(item, str) and item
                        }
                        hit_ids = {
                            item.get("memory_id")
                            for item in hits
                            if isinstance(item, Mapping)
                            and isinstance(item.get("memory_id"), str)
                            and item.get("memory_id")
                        }
                        latest_search_memory_ids = returned & hit_ids
                        latest_read_memory_ids = set()
                    continue
                if operation != "read" or result.get("operation") != "read":
                    continue
                request_arguments = request_value.get("arguments")
                memory = result.get("memory")
                if (
                    isinstance(request_arguments, Mapping)
                    and isinstance(request_arguments.get("memory_id"), str)
                    and request_arguments.get("memory_id")
                    in latest_search_memory_ids
                    and result.get("memory_id")
                    == request_arguments.get("memory_id")
                    and isinstance(memory, Mapping)
                    and memory.get("memory_id")
                    == request_arguments.get("memory_id")
                ):
                    latest_read_memory_ids.add(str(request_arguments["memory_id"]))
            if not latest_search_memory_ids or (
                latest_read_memory_ids != latest_search_memory_ids
            ):
                return "hotpotqa_qa_memory_top_k_not_fully_read"
            if (
                sufficiency == "supported"
                and selected_memory_id not in latest_read_memory_ids
            ):
                return "hotpotqa_qa_memory_selected_memory_has_no_lineage"
        return None


def build_hotpotqa_embedding_tool_registry(
    index: _EmbeddingIndex,
    *,
    task_id: str,
    tool_id: str = HOTPOTQA_RETRIEVAL_TOOL_ID,
    frozen_top_k: int | None = None,
    timeout_seconds: float = 10.0,
) -> ToolRegistry:
    """Build one read-only dynamic retrieval Tool registry.

    ``frozen_top_k`` must be present in the index manifest.  Supplying it here
    is an assertion and cannot override the manifest, which keeps validation
    runs fail-closed against architecture-development configuration drift.
    ``task_id`` scopes the legacy passage index; for global QA-memory it is only
    recorded as the evaluation-call context in Tool results/receipts.
    """

    normalized_task_id = _required_text(task_id, field_name="task_id")
    normalized_tool_id = _required_text(tool_id, field_name="tool_id")
    index_kind = _index_kind(index)
    manifest_top_k = _positive_integer(
        _optional_manifest_field(index.manifest, "frozen_top_k"),
        field_name="retrieval manifest frozen_top_k",
    )
    if frozen_top_k is not None:
        requested_top_k = _positive_integer(
            frozen_top_k, field_name="frozen_top_k"
        )
        if requested_top_k != manifest_top_k:
            raise ValueError("frozen_top_k differs from the index manifest")
    if not isinstance(timeout_seconds, (int, float)) or timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive")

    identity = _index_identity(index, index_kind=index_kind)
    version = str(identity["index_id"])
    search_schema = {
        "title": "search",
        "description": "Arguments for the search Tool action.",
        "type": "object",
        "additionalProperties": False,
        "required": ["query", "k"],
        "properties": {
            "query": {
                "type": "string",
                "minLength": 1,
                "description": "Focused query for missing public evidence.",
            },
            "k": {
                "const": manifest_top_k,
                "description": "Frozen number of embedding-ranked candidates.",
            },
        },
    }
    read_identifier = (
        "memory_id" if _is_qa_memory_index_kind(index_kind) else "doc_id"
    )
    read_schema = {
        "title": "read",
        "description": "Arguments for the read Tool action.",
        "type": "object",
        "additionalProperties": False,
        "required": [read_identifier],
        "properties": {
            read_identifier: {
                "type": "string",
                "minLength": 1,
                "description": (
                    f"Exact {read_identifier} returned by a successful search."
                ),
            }
        },
    }
    hit_schema = {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "doc_id",
            "document_id",
            "title",
            "snippet",
            "similarity",
            "rank",
        ],
        "properties": {
            "doc_id": {"type": "string"},
            "document_id": {"type": "string"},
            "title": {"type": "string"},
            "snippet": {"type": "string"},
            "similarity": {"type": "number"},
            "rank": {"type": "integer", "minimum": 1},
        },
    }
    passage_schema = {
        "type": "object",
        "additionalProperties": False,
        "required": ["doc_id", "document_id", "title", "content"],
        "properties": {
            "doc_id": {"type": "string"},
            "document_id": {"type": "string"},
            "title": {"type": "string"},
            "content": {"type": "string"},
        },
    }
    memory_hit_schema = {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "memory_id",
            "source_train_task_id",
            "paraphrase_question",
            "paraphrase_answer_statement",
            "similarity",
            "rank",
        ],
        "properties": {
            "memory_id": {"type": "string"},
            "source_train_task_id": {"type": "string"},
            "paraphrase_question": {"type": "string"},
            "paraphrase_answer_statement": {"type": "string"},
            "similarity": {"type": "number"},
            "rank": {"type": "integer", "minimum": 1},
        },
    }
    memory_schema = {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "memory_id",
            "source_train_task_id",
            "base_task_id",
            "cycled",
            "paraphrase_question",
            "paraphrase_answer_statement",
            "canonical_answer",
            "paraphrase_version",
            "paraphrase_provenance",
        ],
        "properties": {
            "memory_id": {"type": "string"},
            "source_train_task_id": {"type": "string"},
            "base_task_id": {"type": "string"},
            "cycled": {"type": "boolean"},
            "paraphrase_question": {"type": "string"},
            "paraphrase_answer_statement": {"type": "string"},
            "canonical_answer": {"type": "string"},
            "paraphrase_version": {"type": "string"},
            "paraphrase_provenance": {"type": "string"},
        },
    }
    if _is_qa_memory_index_kind(index_kind):
        output_schema = {
            "oneOf": [
                {
                    "type": "object",
                    "additionalProperties": False,
                    "required": [
                        "operation",
                        "task_id",
                        "retrieval_index",
                        "query",
                        "k",
                        "memory_ids",
                        "hits",
                    ],
                    "properties": {
                        "operation": {"const": "search"},
                        "task_id": {"type": "string"},
                        "retrieval_index": {"type": "object"},
                        "query": {"type": "string"},
                        "k": {"const": manifest_top_k},
                        "memory_ids": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                        "hits": {"type": "array", "items": memory_hit_schema},
                    },
                },
                {
                    "type": "object",
                    "additionalProperties": False,
                    "required": [
                        "operation",
                        "task_id",
                        "retrieval_index",
                        "memory_id",
                        "memory",
                    ],
                    "properties": {
                        "operation": {"const": "read"},
                        "task_id": {"type": "string"},
                        "retrieval_index": {"type": "object"},
                        "memory_id": {"type": "string"},
                        "memory": memory_schema,
                    },
                },
            ]
        }
    else:
        output_schema = {
            "oneOf": [
                {
                    "type": "object",
                    "additionalProperties": False,
                    "required": [
                        "operation",
                        "task_id",
                        "retrieval_index",
                        "query",
                        "k",
                        "doc_ids",
                        "hits",
                    ],
                    "properties": {
                        "operation": {"const": "search"},
                        "task_id": {"type": "string"},
                        "retrieval_index": {"type": "object"},
                        "query": {"type": "string"},
                        "k": {"const": manifest_top_k},
                        "doc_ids": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                        "hits": {"type": "array", "items": hit_schema},
                    },
                },
                {
                    "type": "object",
                    "additionalProperties": False,
                    "required": [
                        "operation",
                        "task_id",
                        "retrieval_index",
                        "doc_id",
                        "passage",
                    ],
                    "properties": {
                        "operation": {"const": "read"},
                        "task_id": {"type": "string"},
                        "retrieval_index": {"type": "object"},
                        "doc_id": {"type": "string"},
                        "passage": passage_schema,
                    },
                },
            ]
        }
    capability = ToolCapability(
        tool_id=normalized_tool_id,
        dataset_scope=HOTPOTQA_DATASET_SCOPE,
        action_schemas={
            "search": search_schema,
            "read": read_schema,
        },
        input_schema={"oneOf": [search_schema, read_schema]},
        output_schema=output_schema,
        side_effect="none",
        timeout_seconds=float(timeout_seconds),
        version=version,
    )
    return ToolRegistry(
        (
            ToolRegistration(
                normalized_tool_id,
                HotpotQAEmbeddingToolBackend(
                    index=index,
                    task_id=normalized_task_id,
                    frozen_top_k=manifest_top_k,
                    index_identity=identity,
                    index_kind=index_kind,
                ),
                capability,
            ),
        )
    )


__all__ = [
    "HOTPOTQA_DATASET_SCOPE",
    "HOTPOTQA_QA_MEMORY_TOOL_ID",
    "HOTPOTQA_RETRIEVAL_TOOL_ID",
    "HotpotQAEmbeddingToolBackend",
    "HotpotQAEmbeddingReactExecutionAdapter",
    "build_hotpotqa_embedding_tool_registry",
]
