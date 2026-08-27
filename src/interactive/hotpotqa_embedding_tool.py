"""HotpotQA embedding retrieval for the shared ToolRegistry.

This is a thin adapter around the public-context embedding index.  It keeps
FlowSteer's ToolRegistry/ReAct boundary and SkillFlow's dynamic search/read
execution shape: retrieval happens only when an Agent dispatches a Tool
action, never as evaluation-time prefetch.

The registry supports the existing task-scoped public-passage index and the
train-only global QA-memory index.  In the passage condition, ``task_id``
selects the current task's public context.  In the QA-memory condition it is
only evaluation-call provenance in the receipt and is never passed to global
``search(query, k)`` or ``read(memory_id)``.  Results are projected through an
explicit corpus-specific allowlist.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from contextvars import ContextVar
from dataclasses import dataclass, field
import inspect
import json
import math
import re
from typing import Protocol
import unicodedata

from .agent_runtime import AgentRequest, GatewayResponse
from .hotpotqa_qa_memory_index import QA_MEMORY_CORPUS_VERSIONS
from .react_execution import ReactExecutionError, ToolReactExecutionAdapter
from .task_dataset import qa_question_scope
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
_QA_MEMORY_EVIDENCE_FIELDS = (
    "question_scope",
    "memory_id",
    "source_train_task_id",
    "paraphrase_question",
    "paraphrase_answer_statement",
    "canonical_answer",
)
_ACTIVE_QA_MEMORY_QUESTION_SCOPE: ContextVar[str | None] = ContextVar(
    "hotpotqa_qa_memory_question_scope",
    default=None,
)


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
    minimum_similarity: float | None = None,
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
            if (
                tool_id == HOTPOTQA_QA_MEMORY_TOOL_ID
                and minimum_similarity is not None
            ):
                raw_hits = result.get("hits", ())
                raw_doc_ids = (
                    tuple(
                        hit.get("memory_id")
                        for hit in raw_hits
                        if isinstance(hit, Mapping)
                        and isinstance(hit.get("memory_id"), str)
                        and isinstance(hit.get("similarity"), (int, float))
                        and not isinstance(hit.get("similarity"), bool)
                        and float(hit["similarity"]) >= minimum_similarity
                    )
                    if isinstance(raw_hits, (list, tuple))
                    else ()
                )
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
        "unique_source_count": ("unique_source_count",),
        "cycled_record_count": ("cycled_record_count",),
        "paraphrase_count": ("paraphrase_count",),
        "heldout_validation_count": ("heldout_validation_count",),
        "validation_overlap_count": ("validation_overlap_count",),
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
    if corpus_version in QA_MEMORY_CORPUS_VERSIONS:
        if _optional_manifest_field(manifest, "source_split", "split") != "train":
            raise ValueError("QA-memory retrieval index must use the train split")
        if _optional_manifest_field(manifest, "validation_overlap_count") != 0:
            raise ValueError("QA-memory retrieval index overlaps held-out validation")
        if search_parameters != ("query", "k") or read_parameters != ("memory_id",):
            raise TypeError("QA-memory manifest and global search/read signatures differ")
        return _QA_MEMORY_INDEX_KIND
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


def _successful_qa_memory_receipt_value(
    receipt: Mapping[str, object],
    *,
    tool_id: str,
    action: str,
) -> tuple[Mapping[str, object], Mapping[str, object]] | None:
    """Return one successful canonical Tool request/result pair.

    DIRECT_REUSE: SkillFlow's Tool receipt is the authoritative boundary for
    an Action--Observation transition.  This helper accepts the persisted
    ``ToolReceipt.to_value`` shape and does not infer evidence from model text.
    """

    request = receipt.get("request")
    result = receipt.get("result")
    if (
        receipt.get("tool_id") != tool_id
        or receipt.get("error_type") is not None
        or not isinstance(request, Mapping)
        or request.get("action") != action
        or not isinstance(result, Mapping)
    ):
        return None
    if "completed" in result and result.get("completed") is not True:
        return None
    value = result.get("value", result)
    if not isinstance(value, Mapping) or value.get("operation") != action:
        return None
    return request, value


def _strict_artifact_text(value: object) -> str | None:
    if not isinstance(value, str) or not value or value != value.strip():
        return None
    return value


def _project_qa_memory_evidence_artifact(
    memory_id: object,
    tool_receipts: Sequence[Mapping[str, object]],
    *,
    question_scope: str,
    tool_id: str = HOTPOTQA_QA_MEMORY_TOOL_ID,
) -> tuple[dict[str, str] | None, str | None]:
    """Project one selected read handle into the native evidence artifact.

    DIRECT_REUSE: SkillFlow treats the successful Tool Observation as the
    authoritative environment value.  The worker policy selects an opaque
    ``memory_id`` through search/read; this adapter then copies the record from
    that exact persisted read receipt instead of asking the model to reproduce
    answer-bearing fields token by token.
    """

    selected_memory_id = _strict_artifact_text(memory_id)
    if selected_memory_id is None:
        return None, "hotpotqa_qa_memory_completion_memory_id_invalid"
    searched_memory_ids: set[str] = set()
    for receipt in tool_receipts:
        if not isinstance(receipt, Mapping):
            continue
        successful_search = _successful_qa_memory_receipt_value(
            receipt,
            tool_id=tool_id,
            action="search",
        )
        if successful_search is not None:
            request, value = successful_search
            arguments = request.get("arguments")
            memory_ids = value.get("memory_ids")
            hits = value.get("hits")
            if (
                isinstance(arguments, Mapping)
                and set(arguments) == {"query", "k"}
                and isinstance(memory_ids, list)
                and isinstance(hits, list)
            ):
                returned_ids = {
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
                searched_memory_ids.update(returned_ids & hit_ids)
            continue

        successful_read = _successful_qa_memory_receipt_value(
            receipt,
            tool_id=tool_id,
            action="read",
        )
        if successful_read is None:
            continue
        request, value = successful_read
        arguments = request.get("arguments")
        memory = value.get("memory")
        if (
            not isinstance(arguments, Mapping)
            or set(arguments) != {"memory_id"}
            or arguments.get("memory_id") != selected_memory_id
            or selected_memory_id not in searched_memory_ids
            or value.get("memory_id") != selected_memory_id
            or not isinstance(memory, Mapping)
            or memory.get("memory_id") != selected_memory_id
        ):
            continue
        projected: dict[str, str] = {"question_scope": question_scope}
        for field_name in _QA_MEMORY_EVIDENCE_FIELDS:
            if field_name == "question_scope":
                continue
            field_value = _strict_artifact_text(memory.get(field_name))
            if field_value is None:
                return (
                    None,
                    f"hotpotqa_qa_memory_read_{field_name}_invalid",
                )
            projected[field_name] = field_value
        return projected, None
    return None, "hotpotqa_qa_memory_artifact_has_no_exact_search_read_lineage"


def _validate_qa_memory_evidence_artifact(
    artifact: str,
    tool_receipts: Sequence[Mapping[str, object]],
    *,
    question_scope: str,
    tool_id: str = HOTPOTQA_QA_MEMORY_TOOL_ID,
) -> tuple[dict[str, str] | None, str | None]:
    """Bind one worker artifact to a prior exact QA-memory search/read pair.

    NECESSARY_ADAPTATION: SkillFlow supplies the canonical search/read receipt
    boundary, while FlowSteer's progressive Canvas transports the resulting
    artifact over explicit AgentGraph relations.  QA-memory records carry a
    structured question/answer payload rather than a public passage, so the
    compatibility gate binds every admitted field to the exact read record.
    """

    try:
        raw_artifact = json.loads(artifact)
    except (TypeError, ValueError):
        return None, "hotpotqa_qa_memory_artifact_not_json_object"
    if not isinstance(raw_artifact, dict):
        return None, "hotpotqa_qa_memory_artifact_not_json_object"
    if set(raw_artifact) != set(_QA_MEMORY_EVIDENCE_FIELDS):
        return None, "hotpotqa_qa_memory_artifact_field_set_invalid"
    fields: dict[str, str] = {}
    for field_name in _QA_MEMORY_EVIDENCE_FIELDS:
        value = _strict_artifact_text(raw_artifact.get(field_name))
        if value is None:
            return None, f"hotpotqa_qa_memory_artifact_{field_name}_invalid"
        fields[field_name] = value
    if fields["question_scope"] != question_scope:
        return None, "hotpotqa_qa_memory_question_scope_mismatch"

    projected, issue = _project_qa_memory_evidence_artifact(
        fields["memory_id"],
        tool_receipts,
        question_scope=question_scope,
        tool_id=tool_id,
    )
    if issue is not None or projected is None:
        return None, issue
    if fields != projected:
        return None, "hotpotqa_qa_memory_artifact_has_no_exact_search_read_lineage"
    return fields, None


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

        if self.index_kind == _QA_MEMORY_INDEX_KIND:
            candidate = self.index.search(query, k)
        else:
            candidate = self.index.search(self.task_id, query, k)
        raw_hits = await candidate if inspect.isawaitable(candidate) else candidate
        if not isinstance(raw_hits, Sequence) or isinstance(raw_hits, (str, bytes)):
            raise TypeError("embedding search must return a sequence of hits")
        if len(raw_hits) > self.frozen_top_k:
            raise ValueError("embedding search returned more than frozen top-k hits")
        if self.index_kind == _QA_MEMORY_INDEX_KIND:
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
            if self.index_kind == _QA_MEMORY_INDEX_KIND
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

        if self.index_kind == _QA_MEMORY_INDEX_KIND:
            candidate = self.index.read(resource_id)
        else:
            candidate = self.index.read(self.task_id, resource_id)
        raw_value = await candidate if inspect.isawaitable(candidate) else candidate
        if self.index_kind == _QA_MEMORY_INDEX_KIND:
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
        qa_memory_min_similarity: float | None = None,
        **kwargs: object,
    ) -> None:
        if qa_memory_min_similarity is not None and (
            isinstance(qa_memory_min_similarity, bool)
            or not isinstance(qa_memory_min_similarity, (int, float))
            or not math.isfinite(float(qa_memory_min_similarity))
            or not 0.0 <= float(qa_memory_min_similarity) <= 1.0
        ):
            raise ValueError(
                "qa_memory_min_similarity must be a finite value in [0, 1]"
            )
        super().__init__(**kwargs)
        self._qa_memory_min_similarity = (
            None
            if qa_memory_min_similarity is None
            else float(qa_memory_min_similarity)
        )

    async def execute(self, request: AgentRequest) -> GatewayResponse:
        """Keep the request-scoped question available to completion admission.

        DIRECT_REUSE: the context-local protocol follows the existing
        SkillFlow-derived request-scoped execution state used by the unified QA
        adapter.  It remains safe when multiple worker executions overlap.
        """

        if self._active_retrieval_tool_id() != HOTPOTQA_QA_MEMORY_TOOL_ID:
            return await super().execute(request)
        token = _ACTIVE_QA_MEMORY_QUESTION_SCOPE.set(
            qa_question_scope(request.problem)
        )
        try:
            try:
                return await super().execute(request)
            except ReactExecutionError as exc:
                metadata = dict(exc.metadata)
                failure_type = self._qa_memory_exhaustion_failure_type(
                    metadata.get("tool_receipts", ()),
                    minimum_similarity=self._qa_memory_min_similarity,
                )
                metadata["retrieval_failure_type"] = failure_type
                trace = list(metadata.get("react_trace", ()))
                trace.append(
                    {
                        "observation_status": "terminal_failure",
                        "public_error_code": failure_type,
                    }
                )
                metadata["react_trace"] = trace
                raise ReactExecutionError(str(exc), metadata=metadata) from exc
        finally:
            _ACTIVE_QA_MEMORY_QUESTION_SCOPE.reset(token)

    @staticmethod
    def _qa_memory_exhaustion_failure_type(
        receipts: object,
        *,
        minimum_similarity: float | None = None,
    ) -> str:
        """Classify a bounded no-answer Tool plan from public receipts only."""

        if not isinstance(receipts, (list, tuple)):
            return "retrieval_strategy_failure"
        successful_searches = 0
        returned_memory_count = 0
        successful_reads = 0
        tool_errors = 0
        for receipt in receipts:
            if not isinstance(receipt, Mapping):
                continue
            if receipt.get("error_type") is not None:
                tool_errors += 1
                continue
            search = _successful_qa_memory_receipt_value(
                receipt,
                tool_id=HOTPOTQA_QA_MEMORY_TOOL_ID,
                action="search",
            )
            if search is not None:
                successful_searches += 1
                value = search[1]
                memory_ids = value.get("memory_ids")
                if isinstance(memory_ids, list):
                    if minimum_similarity is None:
                        returned_memory_count += len(memory_ids)
                    else:
                        hits = value.get("hits", ())
                        if isinstance(hits, list):
                            returned_memory_count += sum(
                                isinstance(hit, Mapping)
                                and isinstance(
                                    hit.get("similarity"), (int, float)
                                )
                                and not isinstance(
                                    hit.get("similarity"), bool
                                )
                                and float(hit["similarity"])
                                >= minimum_similarity
                                for hit in hits
                            )
                continue
            read = _successful_qa_memory_receipt_value(
                receipt,
                tool_id=HOTPOTQA_QA_MEMORY_TOOL_ID,
                action="read",
            )
            if read is not None:
                successful_reads += 1
        if (
            successful_searches > 0
            and returned_memory_count == 0
            and successful_reads == 0
            and tool_errors == 0
        ):
            return "knowledge_base_coverage_failure"
        return "retrieval_strategy_failure"

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
                    "required": ["memory_id"],
                    "additionalProperties": False,
                    "properties": {
                        "memory_id": {
                            "type": "string",
                            "minLength": 1,
                            "description": (
                                "Select one exact memory_id already read in this "
                                "worker execution; the runtime projects the native "
                                "evidence artifact from that Tool receipt."
                            ),
                        }
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
        state = _public_retrieval_state(
            observations,
            tool_id=tool_id,
            minimum_similarity=(
                self._qa_memory_min_similarity
                if tool_id == HOTPOTQA_QA_MEMORY_TOOL_ID
                else None
            ),
        )
        remaining_tool_calls = max(
            0,
            self._max_tool_calls - state.dispatched_tool_calls,
        )
        successful_read_count = len(state.read_doc_ids)
        if successful_read_count >= 2:
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
        if successful_read_count > 0:
            # One missing-hop search/read transition requires two remaining
            # Tool calls. Preserve the first read if a Tool error consumed that
            # capacity instead of admitting an unfinishable blind search.
            if remaining_tool_calls >= 2:
                return frozenset(
                    {(tool_id, "search")}
                ), False
            return frozenset(), True
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
        state = _public_retrieval_state(
            observations,
            tool_id=tool_id,
            minimum_similarity=(
                self._qa_memory_min_similarity
                if tool_id == HOTPOTQA_QA_MEMORY_TOOL_ID
                else None
            ),
        )
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
        elif action_names == {"search"} and state.search_queries:
            next_action = (
                "Current action mask: search only; read and complete are not "
                "admitted. No unread QA-memory candidate met the frozen "
                "embedding confidence threshold for the current entity and "
                "relation, or a prior read left a missing hop. Form a distinct "
                "focused query. A missing mention in one memory is not evidence "
                "that another named entity lacks the requested fact. Prior "
                "successful queries: "
                + json.dumps(
                    list(state.search_queries),
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
                + "."
            )
        elif completion_admitted and not actions:
            next_action = (
                "Current action mask: complete only. Preserve both public read "
                "observations and return their evidence-supported artifact."
            )
        else:
            next_action = (
                "Current action mask: search only; read and complete are not "
                "admitted. Supply all required search arguments."
            )
        qa_memory_completion = ""
        completion_example = '{"value":"evidence-supported artifact"}'
        if tool_id == HOTPOTQA_QA_MEMORY_TOOL_ID:
            completion_example = json.dumps(
                {"value": {"memory_id": "exact read memory_id"}},
                ensure_ascii=False,
                separators=(",", ":"),
            )
            qa_memory_completion = (
                " For train QA-memory, completion value contains only the exact "
                "memory_id selected by a successful read in this worker execution. "
                "The runtime deterministically projects the native evidence "
                "artifact from that persisted Tool receipt; do not copy or invent "
                "answer-bearing record fields."
            )
        return (
            base
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
            "repeat a normalized successful query. After the first successful "
            "read, use the next search/read transition for the missing entity, "
            "relation, or hop while the frozen Tool budget remains. Complete "
            "only on the admitted terminal state, with exactly the declared "
            "completion shape "
            + completion_example
            + ". "
            + next_action
            + qa_memory_completion
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
            minimum_similarity=(
                self._qa_memory_min_similarity
                if self._active_retrieval_tool_id()
                == HOTPOTQA_QA_MEMORY_TOOL_ID
                else None
            ),
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
        del action
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
            question_scope = _ACTIVE_QA_MEMORY_QUESTION_SCOPE.get()
            if question_scope is None:
                return "hotpotqa_qa_memory_question_scope_unavailable"
            try:
                selection = json.loads(artifact)
            except (TypeError, ValueError):
                return "hotpotqa_qa_memory_completion_selection_invalid"
            if not isinstance(selection, dict) or set(selection) != {"memory_id"}:
                return "hotpotqa_qa_memory_completion_selection_invalid"
            _, issue = _project_qa_memory_evidence_artifact(
                selection.get("memory_id"),
                tool_receipts,
                question_scope=question_scope,
                tool_id=tool_id,
            )
            return issue
        return None

    def _completion_artifact(
        self,
        *,
        action: object,
        artifact: str,
        tool_receipts: list[dict[str, object]],
    ) -> str:
        if self._active_retrieval_tool_id() != HOTPOTQA_QA_MEMORY_TOOL_ID:
            return super()._completion_artifact(
                action=action,
                artifact=artifact,
                tool_receipts=tool_receipts,
            )
        question_scope = _ACTIVE_QA_MEMORY_QUESTION_SCOPE.get()
        selection = json.loads(artifact)
        projected, issue = _project_qa_memory_evidence_artifact(
            selection.get("memory_id") if isinstance(selection, Mapping) else None,
            tool_receipts,
            question_scope=question_scope or "",
            tool_id=HOTPOTQA_QA_MEMORY_TOOL_ID,
        )
        if issue is not None or projected is None:
            # ``_completion_error`` runs immediately before this hook.  Keep a
            # fail-closed guard in case a future caller bypasses that boundary.
            return ""
        return json.dumps(
            projected,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )


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
    ``task_id`` scopes the legacy passage index; for train QA-memory it is only
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
        "memory_id" if index_kind == _QA_MEMORY_INDEX_KIND else "doc_id"
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
            "similarity",
            "rank",
        ],
        "properties": {
            "memory_id": {"type": "string"},
            "source_train_task_id": {"type": "string"},
            "paraphrase_question": {"type": "string"},
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
    if index_kind == _QA_MEMORY_INDEX_KIND:
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
