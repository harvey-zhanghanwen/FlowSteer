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
from contextvars import ContextVar
from dataclasses import dataclass, field
import inspect
import json
from pathlib import Path
import re
from tempfile import TemporaryDirectory
from typing import Callable, Mapping, Protocol, Sequence

from .agent_runtime import AgentGateway, AgentRequest, GatewayResponse
from .react_execution import ToolReactExecutionAdapter
from .qa_retrieval import (
    DEFAULT_QA_RETRIEVAL_INDEX,
    DEFAULT_SKILLFLOW_SOURCE,
    SkillFlowRetrievalError,
    _load_retrieval_index_class,
    _load_retrieval_module,
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
_PROVIDED_PASSAGE = re.compile(
    r"^\[(?P<title>[^\]]+)\]\s*(?P<text>.+)$",
    flags=re.DOTALL,
)

SKILLFLOW_MULTI_HOP_QA_GUIDANCE = (
    "Chain evidence across passages to answer multi-hop questions. "
    "Search for specific entity names (not the full question). "
    "If a search has no match or repeats prior evidence, pivot with synonyms. "
    "Complete with a brief name, date, number, or short phrase."
)
SKILLFLOW_FACTUAL_QA_GUIDANCE = (
    "Extract the answer from retrieved passages and do not guess from memory. "
    "Complete with a concise name, place, number, or short phrase."
)
HOTPOTQA_VERIFIED_ANSWER_SLOT_GUIDANCE = (
    "Treat ReAct only as the execution schedule Thought -> Action(tool) -> "
    "Observation -> Thought -> Final, never as an Agent role. Preserve the "
    "question's exact scope and answer slot. Represent retrieved facts as "
    "subject/entity, predicate/relation, object or attribute value, and qualifiers; "
    "keep the entity-to-attribute binding explicit and show every bridge in the "
    "multi-hop chain. If compared values are unexpectedly equal, recheck scope, "
    "both bindings, retrieved passages, and contract narrowing before calling a tie."
)


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

    def __init__(
        self,
        *,
        gateway: AgentGateway,
        tool_registry: ToolRegistry,
        max_turns: int,
        max_tool_calls: int,
        max_action_tokens: int = 512,
        task_type: str | None = None,
        completion_policy: str = "required_tool_call",
    ) -> None:
        if task_type not in {None, "multi_hop_qa", "factual_qa"}:
            raise ValueError("QA task_type must be multi_hop_qa, factual_qa, or None")
        if completion_policy not in {
            "optional",
            "required_tool_call",
            "required_evidence",
        }:
            raise ValueError(
                "QA completion_policy must be optional, required_tool_call, "
                "or required_evidence"
            )
        super().__init__(
            gateway=gateway,
            tool_registry=tool_registry,
            max_turns=max_turns,
            max_tool_calls=max_tool_calls,
            max_action_tokens=max_action_tokens,
        )
        self._task_type = task_type
        self._completion_policy = completion_policy
        self._retrieval_completion_required: ContextVar[bool] = ContextVar(
            f"qa_retrieval_completion_required_{id(self)}",
            default=False,
        )

    def _required_evidence_state(
        self,
        request: AgentRequest,
        observations: list[Mapping[str, object]],
    ) -> tuple[bool, tuple[str, ...], bool]:
        required = (
            self._completion_policy == "required_evidence"
            and self._max_tool_calls > 0
            and QA_RETRIEVAL_TOOL_ID in request.agent.allowed_tools
        )
        searched_passage_ids: list[str] = []
        has_successful_read = False
        if required:
            for observation in observations:
                if observation.get("observation_status") != "success":
                    continue
                result = observation.get("result")
                if not isinstance(result, Mapping):
                    continue
                if result.get("operation") == "search":
                    raw_ids = result.get("passage_ids")
                    if isinstance(raw_ids, list):
                        searched_passage_ids = [
                            value.strip()
                            for value in raw_ids
                            if isinstance(value, str) and value.strip()
                        ]
                elif (
                    result.get("operation") == "read"
                    and isinstance(result.get("passage"), Mapping)
                    and isinstance(result["passage"].get("text"), str)
                    and bool(result["passage"]["text"].strip())
                ):
                    has_successful_read = True
        return required, tuple(searched_passage_ids), has_successful_read

    def _state_conditioned_action_domain(
        self,
        request: AgentRequest,
        observations: list[Mapping[str, object]],
    ) -> tuple[Optional[frozenset[tuple[str, str]]], bool]:
        """Expose the canonical search -> read -> complete action domain."""

        required, passage_ids, has_successful_read = self._required_evidence_state(
            request,
            observations,
        )
        if not required:
            return super()._state_conditioned_action_domain(request, observations)
        if has_successful_read:
            return frozenset(), True
        action_name = "read" if passage_ids else "search"
        return frozenset({(QA_RETRIEVAL_TOOL_ID, action_name)}), False

    async def execute(self, request: AgentRequest) -> GatewayResponse:
        # NECESSARY_ADAPTATION: the generic AgentGraph completion hook receives
        # Tool receipts but not the current AgentRequest.  Bind only the QA
        # request-scoped admission bit here so concurrent executions cannot
        # leak allowed-tool state across Agents.  A zero Tool budget preserves
        # the generic direct-completion boundary because no dispatch is legal.
        requires_retrieval = (
            self._completion_policy != "optional"
            and self._max_tool_calls > 0
            and QA_RETRIEVAL_TOOL_ID in request.agent.allowed_tools
        )
        token = self._retrieval_completion_required.set(requires_retrieval)
        try:
            return await super().execute(request)
        finally:
            self._retrieval_completion_required.reset(token)

    def _contract(
        self,
        request: AgentRequest,
        observations: list[Mapping[str, object]],
    ) -> str:
        contract = super()._contract(request, observations)
        required_evidence, searched_passage_ids, has_successful_read = (
            self._required_evidence_state(request, observations)
        )
        completion_admitted = not required_evidence or has_successful_read
        if self._task_type == "multi_hop_qa":
            guidance = SKILLFLOW_MULTI_HOP_QA_GUIDANCE
            if request.semantic_protocol == "hotpotqa_verified_answer_slot_v1":
                guidance += " " + HOTPOTQA_VERIFIED_ANSWER_SLOT_GUIDANCE
        elif self._task_type == "factual_qa":
            guidance = SKILLFLOW_FACTUAL_QA_GUIDANCE
        else:
            guidance = ""
        # DIRECT_REUSE: SkillFlow training/task_prompts.py::{MULTI_HOP_QA,
        # FACTUAL_QA}.  Only the terminal wire is adapted from answer(response=)
        # to this runtime's already-declared StructuredAction completion.
        terminal_wire = (
            " On completion, arguments.value is the completed QA artifact, "
            "not a schema label or placeholder."
            if completion_admitted
            else (
                " Completion is not admitted in this Tool-only state; preserve "
                "the eventual artifact responsibility but emit only the current "
                "Tool action and its declared arguments."
            )
        )
        semantic_role = (request.agent.role_family or "").casefold()
        if (
            request.semantic_protocol == "hotpotqa_verified_answer_slot_v1"
            and semantic_role == "reasoner"
        ):
            if completion_admitted:
                terminal_wire += (
                    " As the Reasoner, arguments.value must contain the labeled "
                    "Question scope, Answer slot, Evidence propositions, Multi-hop "
                    "chain, Candidate answer, and Evidence fields. Determine the one "
                    "semantic candidate only after aligning the evidence proposition "
                    "to the requested answer slot."
                )
            else:
                terminal_wire += (
                    " The Reasoner remains responsible for semantic alignment after "
                    "evidence is read, but no semantic-answer field belongs in the "
                    "current Tool arguments."
                )
        elif (
            request.semantic_protocol == "hotpotqa_verified_answer_slot_v1"
            and semantic_role == "verifier"
        ):
            terminal_wire += (
                " As the Verifier, check explicit retrieved evidence, entity-to-"
                "attribute binding, multi-hop completeness, and unchanged question "
                "scope. Do not replace the Reasoner's candidate. Return Candidate "
                "answer, the four explicit boolean check fields, and Verification "
                "status. Set supported only when all checks pass; "
                "otherwise request repair without a substitute candidate."
            )
        elif request.is_format_predecessor:
            terminal_wire += (
                " As the direct predecessor of the Format Agent, put the brief "
                "answer span after `Candidate answer:` and the supporting "
                "retrieved passage span after `Evidence:` in arguments.value."
            )
        evidence_continuation = ""
        if (
            self._completion_policy == "required_evidence"
            and QA_RETRIEVAL_TOOL_ID in request.agent.allowed_tools
        ):
            evidence_continuation = (
                "\nRequired-evidence ReAct continuation: preserve any semantic "
                "candidate already present in the public action history; do not "
                "discard or replace it merely because completion was rejected. "
            )
            if has_successful_read:
                evidence_continuation += (
                    "A successful non-empty qa-retrieval read is present, so the "
                    "next action may complete after aligning that evidence to the "
                    "original answer slot. This turn use kind=complete, "
                    "name=complete, resource_id=null, and skill_id=null; arguments "
                    "must contain exactly one key, value, whose value is the full "
                    "labeled semantic artifact. Do not use kind=completion, "
                    "name=answer, or resource_id=qa-retrieval."
                )
            elif searched_passage_ids:
                read_wire = {
                    "arguments": {"passage_id": searched_passage_ids[0]},
                    "kind": "tool",
                    "name": "read",
                    "resource_id": QA_RETRIEVAL_TOOL_ID,
                    "skill_id": None,
                }
                evidence_continuation += (
                    "The next action must be qa-retrieval read using one exact "
                    "passage_id returned by the successful search observation: "
                    + json.dumps(searched_passage_ids, ensure_ascii=False)
                    + ". Its arguments object contains only passage_id; never put "
                    "Question scope, Answer slot, Evidence propositions, Multi-hop "
                    "chain, Candidate answer, Evidence, JSON-Schema properties, or "
                    "additionalProperties into read arguments. One valid exact wire "
                    "using the first ranked returned passage is: "
                    + json.dumps(
                        read_wire,
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    )
                    + ". Do not call complete before that read succeeds."
                )
            else:
                evidence_continuation += (
                    "The next action must be qa-retrieval search with a concise "
                    "entity-and-relation query. Then read an exact returned "
                    "passage_id. Search arguments contain exactly query and limit; "
                    "never copy JSON-Schema properties/additionalProperties or the "
                    "eventual semantic artifact into those arguments. Set query to "
                    "one non-empty focused entity-and-relation string selected from "
                    "the original question and limit to one positive integer. Keep "
                    "the outer constants kind=tool, name=search, "
                    "resource_id=qa-retrieval, and skill_id=null. Do not call "
                    "complete before a non-empty read succeeds."
                )
        qa_guidance = (
            "\nSkillFlow QA execution guidance: " + guidance + terminal_wire
            if guidance
            else ""
        )
        return contract + qa_guidance + evidence_continuation

    def _completion_error(
        self,
        *,
        action: StructuredAction,
        artifact: str,
        tool_receipts: list[dict[str, object]],
    ) -> str | None:
        del action, artifact
        if not self._retrieval_completion_required.get():
            return None

        # DIRECT_REUSE: SkillFlow training/environment.py::step admits the
        # terminal answer after at least one non-answer Tool turn.  Preserve
        # that historical ``required_tool_call`` boundary, including a failed
        # receipt, so a Tool outage cannot discard a usable upstream answer.
        # ``required_evidence`` is an explicit stricter experiment condition:
        # only a successful read receipt carrying non-empty public text counts.
        for receipt in tool_receipts:
            if receipt.get("tool_id") != QA_RETRIEVAL_TOOL_ID:
                continue
            request = receipt.get("request")
            if not isinstance(request, Mapping) or request.get("action") not in {
                "search",
                "read",
            }:
                continue
            if self._completion_policy == "required_tool_call":
                return None
            result = receipt.get("result")
            if not isinstance(result, Mapping) or receipt.get("error_type") is not None:
                continue
            value = result.get("value")
            if not isinstance(value, Mapping):
                continue
            has_evidence = (
                value.get("operation") == "read"
                and isinstance(value.get("passage"), Mapping)
                and isinstance(value["passage"].get("text"), str)
                and bool(value["passage"]["text"].strip())
            )
            if has_evidence:
                return None
        if self._completion_policy == "required_tool_call":
            return "qa_completion_requires_retrieval_dispatch"
        return "qa_completion_requires_successful_read_evidence"

    def _tool_action_error(
        self,
        *,
        request: AgentRequest,
        action: StructuredAction,
        observations: list[Mapping[str, object]],
    ) -> str | None:
        required, passage_ids, has_successful_read = self._required_evidence_state(
            request,
            observations,
        )
        if required and action.kind is ActionKind.TOOL:
            if has_successful_read:
                return "qa_required_evidence_next_action_complete"
            expected_action = "read" if passage_ids else "search"
            if (
                action.resource_id != QA_RETRIEVAL_TOOL_ID
                or action.name != expected_action
            ):
                return f"qa_required_evidence_next_action_{expected_action}"
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
    _cleanup: Callable[[], None] | None = field(default=None, repr=False)
    _closed: bool = field(default=False, init=False, repr=False)

    def close(self) -> None:
        if not self._closed:
            try:
                self._index.close()
            finally:
                self._closed = True
                if self._cleanup is not None:
                    self._cleanup()

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


def _provided_context_passages(
    values: Sequence[str],
    *,
    document_passage_class: object,
) -> tuple[object, ...]:
    if isinstance(values, (str, bytes)) or not values:
        raise ValueError("provided QA context must be a non-empty passage sequence")
    passages: list[object] = []
    for index, value in enumerate(values):
        if not isinstance(value, str) or not value.strip():
            raise ValueError("provided QA context contains an empty passage")
        normalized = value.strip()
        matched = _PROVIDED_PASSAGE.fullmatch(normalized)
        title = (
            matched.group("title").strip()
            if matched is not None
            else f"Provided passage {index + 1}"
        )
        text = matched.group("text").strip() if matched is not None else normalized
        passages.append(
            document_passage_class(
                passage_id=f"provided-context-{index + 1:02d}",
                document_id=f"provided-document-{index + 1:02d}",
                title=title,
                text=text,
            )
        )
    return tuple(passages)


def open_provided_context_qa_tool_registry(
    passages: Sequence[str],
    *,
    skillflow_source: str | Path = DEFAULT_SKILLFLOW_SOURCE,
    dataset_scope: Sequence[str] = ("hotpotqa",),
    timeout_seconds: float = 10.0,
) -> OpenQAToolRegistry:
    """Build and open SkillFlow's FTS index over one task's supplied context.

    SkillFlow ``GenericTaskEnvironment._search_passages`` searches ``context``
    before its external corpus.  The free-AgentGraph runtime uses SkillFlow's
    newer public ``DocumentPassage -> build_retrieval_index -> RetrievalIndex``
    boundary to preserve that behavior without implementing another ranker.
    """

    module = _load_retrieval_module(Path(skillflow_source))
    try:
        document_passage_class = module.DocumentPassage
        build_retrieval_index = module.build_retrieval_index
        retrieval_index_class = module.RetrievalIndex
    except AttributeError as exc:
        raise SkillFlowRetrievalError(
            "SkillFlow task-context retrieval components are unavailable"
        ) from exc

    normalized_passages = _provided_context_passages(
        passages,
        document_passage_class=document_passage_class,
    )
    temporary = TemporaryDirectory(prefix="flowsteer-provided-qa-context-")
    index_path = Path(temporary.name) / "retrieval.sqlite3"
    try:
        build_retrieval_index(
            index_path,
            normalized_passages,
            corpus_name="benchmark-provided-context",
            corpus_version="task-scoped-v1",
        )
        index = _ThreadAffineRetrievalWorker(
            retrieval_index_class,
            index_path,
        )
        registry = build_qa_tool_registry(
            index,
            dataset_scope=dataset_scope,
            timeout_seconds=timeout_seconds,
        )
        identity = _index_identity(index)
    except BaseException:
        temporary.cleanup()
        raise
    return OpenQAToolRegistry(
        registry,
        identity,
        index,
        _cleanup=temporary.cleanup,
    )


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
    "open_provided_context_qa_tool_registry",
]
