"""Task-scoped HotpotQA embedding retrieval for the shared ToolRegistry.

This is a thin adapter around the public-context embedding index.  It keeps
FlowSteer's ToolRegistry/ReAct boundary and SkillFlow's dynamic search/read
execution shape: retrieval happens only when an Agent dispatches a Tool
action, never as evaluation-time prefetch.

The registry binds one HotpotQA ``task_id`` at construction time.  Agent
actions therefore cannot select another task's corpus.  Results are projected
through a public-field allowlist so answers, supporting-fact labels, evaluator
state, and other private metadata cannot enter Tool observations or receipts.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import inspect
import json
import math
from typing import Protocol

from .agent_runtime import AgentRequest
from .react_execution import ToolReactExecutionAdapter
from .tool_runtime import (
    ToolCapability,
    ToolRegistration,
    ToolRegistry,
    ToolRequest,
    ToolResult,
)


HOTPOTQA_RETRIEVAL_TOOL_ID = "qa-retrieval"
HOTPOTQA_DATASET_SCOPE = ("hotpotqa",)


class _TaskScopedEmbeddingIndex(Protocol):
    """Public contract implemented by ``HotpotQAEmbeddingIndex``."""

    @property
    def manifest(self) -> object:
        ...

    def search(
        self,
        task_id: str,
        query: str,
        k: int,
    ) -> Sequence[object]:
        ...

    def read(self, task_id: str, doc_id: str) -> object:
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


def _index_identity(index: _TaskScopedEmbeddingIndex) -> dict[str, object]:
    """Return only public, replay-relevant manifest fields."""

    manifest = index.manifest
    index_id = _required_text(
        _optional_manifest_field(manifest, "index_id"),
        field_name="retrieval manifest index_id",
    )
    identity: dict[str, object] = {"index_id": index_id}
    aliases = {
        "corpus_version": ("corpus_version",),
        "source": ("source", "source_dataset"),
        "split": ("split", "source_split"),
        "embedding_model": ("embedding_model", "embedding_model_id"),
        "embedding_dimension": ("embedding_dimension", "dimension"),
        "normalized": ("normalized", "embedding_normalized"),
        "similarity": ("similarity", "similarity_metric"),
        "document_count": ("document_count", "doc_count"),
        "passage_count": ("passage_count",),
        "frozen_top_k": ("frozen_top_k",),
    }
    for public_name, candidate_names in aliases.items():
        raw = _optional_manifest_field(manifest, *candidate_names)
        if raw is not None:
            identity[public_name] = raw
    return identity


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


@dataclass(slots=True)
class HotpotQAEmbeddingToolBackend:
    """Dispatch task-bound embedding ``search`` and evidence ``read``."""

    index: _TaskScopedEmbeddingIndex
    task_id: str
    frozen_top_k: int
    index_identity: Mapping[str, object]
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

        candidate = self.index.search(self.task_id, query, k)
        raw_hits = await candidate if inspect.isawaitable(candidate) else candidate
        if not isinstance(raw_hits, Sequence) or isinstance(raw_hits, (str, bytes)):
            raise TypeError("embedding search must return a sequence of hits")
        if len(raw_hits) > self.frozen_top_k:
            raise ValueError("embedding search returned more than frozen top-k hits")
        hits = [
            _public_hit(raw_hit, expected_rank=rank)
            for rank, raw_hit in enumerate(raw_hits, start=1)
        ]
        doc_ids = [str(hit["doc_id"]) for hit in hits]
        if len(set(doc_ids)) != len(doc_ids):
            raise ValueError("embedding search returned duplicate doc_id values")
        self._search_returned_doc_ids.update(doc_ids)
        return ToolResult(
            {
                "operation": "search",
                "task_id": self.task_id,
                "retrieval_index": dict(self.index_identity),
                "query": query,
                "k": k,
                "doc_ids": doc_ids,
                "hits": hits,
            }
        )


    async def _read(self, request: ToolRequest) -> ToolResult:
        _validate_action(request, "read")
        if set(request.arguments) != {"doc_id"}:
            raise ValueError("read arguments must contain exactly doc_id")
        doc_id = _required_text(
            request.arguments["doc_id"], field_name="read doc_id"
        )
        if doc_id not in self._search_returned_doc_ids:
            raise ValueError("read doc_id was not returned by a successful search")

        candidate = self.index.read(self.task_id, doc_id)
        raw_passage = (
            await candidate if inspect.isawaitable(candidate) else candidate
        )
        returned_passage_id = _required_text(
            _field(raw_passage, "passage_id"),
            field_name="read passage passage_id",
        )
        if returned_passage_id != doc_id:
            raise ValueError("embedding read returned a different passage_id")
        public_passage = {
            "doc_id": returned_passage_id,
            "document_id": _required_text(
                _field(raw_passage, "document_id"),
                field_name="read passage document_id",
            ),
            "title": _required_text(
                _field(raw_passage, "title"), field_name="read passage title"
            ),
            "content": _required_text(
                _field(raw_passage, "text"), field_name="read passage text"
            ),
        }
        return ToolResult(
            {
                "operation": "read",
                "task_id": self.task_id,
                "retrieval_index": dict(self.index_identity),
                "doc_id": doc_id,
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

    def _allowed_tools_for_turn(
        self,
        request: AgentRequest,
        observations: list[Mapping[str, object]],
    ) -> tuple[str, ...]:
        del request
        for observation in observations:
            result = observation.get("result")
            if (
                observation.get("observation_status") == "success"
                and isinstance(result, Mapping)
                and result.get("operation") == "read"
            ):
                return ()
        return (HOTPOTQA_RETRIEVAL_TOOL_ID,)

    def _contract(
        self,
        request: AgentRequest,
        observations: list[Mapping[str, object]],
    ) -> str:
        base = super()._contract(request, observations)
        capability = self._tool_registry.require_capability(
            HOTPOTQA_RETRIEVAL_TOOL_ID
        )
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
        returned_doc_ids: list[str] = []
        successful_read = False
        for observation in observations:
            if observation.get("observation_status") != "success":
                continue
            result = observation.get("result")
            if not isinstance(result, Mapping):
                continue
            if result.get("operation") == "search":
                raw_doc_ids = result.get("doc_ids", ())
                if isinstance(raw_doc_ids, (list, tuple)):
                    returned_doc_ids.extend(
                        str(item) for item in raw_doc_ids if isinstance(item, str)
                    )
            elif result.get("operation") == "read":
                successful_read = True
        if successful_read:
            return (
                request.agent.contract
                + "\n\nA public passage read has succeeded and Tool execution is "
                "complete. Return exactly one StructuredAction JSON object and "
                "no other text: "
                "{\"arguments\":{\"value\":\"evidence-supported artifact\"},"
                "\"kind\":\"complete\",\"name\":\"complete\","
                "\"resource_id\":null,\"skill_id\":null}. Replace only the "
                "value string with the artifact required by the completion "
                "condition. Public observations: "
                + json.dumps(
                    [dict(observation) for observation in observations],
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
            )
        if returned_doc_ids:
            next_action = (
                "Legal next action: read only. Search is no longer admissible. "
                "Use exactly one of these returned doc_id values: "
                + json.dumps(
                    list(dict.fromkeys(returned_doc_ids)),
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
                + "."
            )
        else:
            next_action = (
                "Legal next action: search only, with all required arguments."
            )
        return (
            base
            + "\nHotpotQA retrieval protocol: search arguments must be exactly "
            + json.dumps(
                {"query": "focused evidence query", "k": frozen_k},
                sort_keys=True,
                separators=(",", ":"),
            )
            + ". After a successful search, choose an exact returned doc_id and "
            "call read with exactly {\"doc_id\":\"returned doc_id\"}. Do not "
            "repeat an identical successful search. After a successful read, "
            "complete with exactly {\"value\":\"evidence-supported artifact\"}. "
            + next_action
        )

    def _completion_error(
        self,
        *,
        action: object,
        artifact: str,
        tool_receipts: list[dict[str, object]],
    ) -> str | None:
        del action, artifact
        successful_actions = {
            str(request["action"])
            for receipt in tool_receipts
            if receipt.get("tool_id") == HOTPOTQA_RETRIEVAL_TOOL_ID
            and receipt.get("error_type") is None
            and isinstance(receipt.get("result"), Mapping)
            and isinstance((request := receipt.get("request")), Mapping)
            and request.get("action") in {"search", "read"}
        }
        if "search" not in successful_actions:
            return "hotpotqa_dynamic_search_required"
        if "read" not in successful_actions:
            return "hotpotqa_dynamic_read_required"
        return None


def build_hotpotqa_embedding_tool_registry(
    index: _TaskScopedEmbeddingIndex,
    *,
    task_id: str,
    frozen_top_k: int | None = None,
    timeout_seconds: float = 10.0,
) -> ToolRegistry:
    """Build one task-bound, read-only dynamic retrieval Tool registry.

    ``frozen_top_k`` must be present in the index manifest.  Supplying it here
    is an assertion and cannot override the manifest, which keeps validation
    runs fail-closed against architecture-development configuration drift.
    """

    normalized_task_id = _required_text(task_id, field_name="task_id")
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

    identity = _index_identity(index)
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
                "description": "Frozen number of embedding-ranked passages.",
            },
        },
    }
    read_schema = {
        "title": "read",
        "description": "Arguments for the read Tool action.",
        "type": "object",
        "additionalProperties": False,
        "required": ["doc_id"],
        "properties": {
            "doc_id": {
                "type": "string",
                "minLength": 1,
                "description": "Exact doc_id returned by a successful search.",
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
    capability = ToolCapability(
        tool_id=HOTPOTQA_RETRIEVAL_TOOL_ID,
        dataset_scope=HOTPOTQA_DATASET_SCOPE,
        input_schema={"oneOf": [search_schema, read_schema]},
        output_schema={
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
        },
        side_effect="none",
        timeout_seconds=float(timeout_seconds),
        version=version,
    )
    return ToolRegistry(
        (
            ToolRegistration(
                HOTPOTQA_RETRIEVAL_TOOL_ID,
                HotpotQAEmbeddingToolBackend(
                    index=index,
                    task_id=normalized_task_id,
                    frozen_top_k=manifest_top_k,
                    index_identity=identity,
                ),
                capability,
            ),
        )
    )


__all__ = [
    "HOTPOTQA_DATASET_SCOPE",
    "HOTPOTQA_RETRIEVAL_TOOL_ID",
    "HotpotQAEmbeddingToolBackend",
    "HotpotQAEmbeddingReactExecutionAdapter",
    "build_hotpotqa_embedding_tool_registry",
]
