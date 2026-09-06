"""Source-separated retrieval using SkillFlow's existing public-document index.

This is a task-specific adapter, not a shared Agent memory or Skill store.
Each invocation sees only the original conversation, its own retained Tool
receipts and receipts delivered by declared graph edges. The existing ReAct
loop, Tool budgets, Canvas scheduling and evaluator remain authoritative.
"""
from __future__ import annotations

import asyncio
from contextvars import ContextVar
from dataclasses import dataclass, replace
import json
from pathlib import Path
from typing import Mapping

from .agent_runtime import AgentRequest, AgentResponse, CommunicationCondition
from .healthbench_clinical_react import HealthBenchClinicalReactExecutionAdapter
from .healthbench_clinical_tools import open_healthbench_clinical_tool_registry
from .healthbench_knowledge_store import HealthBenchKnowledgeStore
from .healthbench_professional_adapter import parse_model_visible_conversation
from .openai_gateway import _healthbench_search_candidates
from .qa_retrieval import DEFAULT_SKILLFLOW_SOURCE
from .tool_runtime import ToolCapability, ToolRegistration, ToolRegistry, ToolRequest, ToolResult

HEALTHBENCH_KNOWLEDGE_TOOL_ID = "healthbench-knowledge.search"
HEALTHBENCH_KNOWLEDGE_VERSION = "skillflow.source-separated-fts5.v1"
_CURRENT_STORE: ContextVar[HealthBenchKnowledgeStore | None] = ContextVar(
    "healthbench_request_knowledge_store", default=None,
)
_COUNT_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "required": ["conversation", "medical_references", "drug_labels"],
    "properties": {name: {"type": "integer", "minimum": 0}
                   for name in ("conversation", "medical_references", "drug_labels")},
}
_INDEX_STATUS_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "required": ["status", "counts", "error_types", "directory"],
    "properties": {
        "status": {"enum": ["available", "partially_indexed", "unavailable"]},
        "counts": {"anyOf": [_COUNT_SCHEMA, {"type": "null"}]},
        "error_types": {"type": "array", "items": {"type": "string"}},
        "directory": {"type": ["string", "null"]},
    },
}


def _ingest_receipts(store, receipts):
    errors = []
    for _, _, evidence in _healthbench_search_candidates(receipts):
        try:
            store.ingest_evidence(evidence)
        except (TypeError, ValueError, OSError, RuntimeError) as exc:
            # Failed indexing must not discard an already obtained source.
            errors.append(type(exc).__name__)
    return errors


def _routed_receipts(request: AgentRequest):
    """Walk only existing public envelopes, never the global graph's outputs.

    Reuse the V3 receipt projection's bounded envelope traversal. Do not index
    the Agent's free-text summary; those statements are not published evidence.
    """
    yield from request.prior_tool_receipts
    if request.communication_condition is CommunicationCondition.UPSTREAM_MASKED:
        return
    pending = [item.to_dict() for item in request.upstream]
    if request.peer_draft is not None:
        pending.append(request.peer_draft.to_dict())
    seen = set()
    examined = 0
    while pending and examined < 16:
        item = pending.pop(0)
        identity = (item.get("source_agent_id"), item.get("artifact_version"),
                    item.get("graph_revision"), item.get("content", item.get("artifact", "")))
        if identity in seen:
            continue
        seen.add(identity)
        examined += 1
        yield from (r for r in item.get("tool_receipts", ()) if isinstance(r, Mapping))
        pending.extend(r for r in item.get("input_artifact_provenance", ()) if isinstance(r, Mapping))


@dataclass(frozen=True, slots=True)
class _IndexingToolBackend:
    original: ToolRegistry
    tool_id: str

    async def invoke(self, request: ToolRequest) -> ToolResult:
        # DIRECT_REUSE: original backend, availability and timeout; no extra
        # network request, query rewriting, synthesized source or reward logic.
        result = await self.original.ainvoke(self.tool_id, request)
        if not isinstance(result.value, dict):
            return result
        store = _CURRENT_STORE.get()
        errors = []
        if store is not None and result.completed:
            errors = await asyncio.to_thread(_ingest_receipts, store, [{
                "tool_id": self.tool_id, "request": request.to_value(),
                "result": result.to_value(), "error_type": None,
            }])
        status = {"status": "unavailable" if store is None else "partially_indexed" if errors else "available",
                  "counts": None if store is None else store.counts(), "error_types": errors,
                  "directory": None if store is None else str(store.directory)}
        return ToolResult({**result.value, "knowledge_index": status}, completed=result.completed)


class _KnowledgeSearchBackend:
    def invoke(self, request: ToolRequest) -> ToolResult:
        if request.action != "search" or set(request.arguments) != {"database", "query"}:
            raise ValueError("knowledge search requires database and query")
        store = _CURRENT_STORE.get()
        if store is None:
            raise RuntimeError("knowledge database is unavailable for this Agent invocation")
        result = store.search(request.arguments["database"], request.arguments["query"], limit=3)
        return ToolResult({**result, "operation": "search", "query": request.arguments["query"],
                           "database_counts": store.counts()})


def build_healthbench_knowledge_tool_registry(original: ToolRegistry) -> ToolRegistry:
    """Wrap the existing five tools and register one classified local search."""
    registrations = []
    for tool_id in original.resource_ids:
        capability = original.require_capability(tool_id)
        output_schema = dict(capability.output_schema)
        output_schema["properties"] = {**output_schema.get("properties", {}),
                                       "knowledge_index": _INDEX_STATUS_SCHEMA}
        registrations.append(ToolRegistration(tool_id, _IndexingToolBackend(original, tool_id),
                            replace(capability, output_schema=output_schema,
                                    version=capability.version + "+" + HEALTHBENCH_KNOWLEDGE_VERSION)))
    arguments = {
        "type": "object", "additionalProperties": False,
        "required": ["database", "query"],
        "properties": {
            "database": {"enum": ["conversation", "medical_references", "drug_labels"],
                         "description": "Conversation is supplied statements, not verified medical evidence. External databases contain only sources observed by this node or routed by graph edges."},
            "query": {"type": "string", "minLength": 1, "maxLength": 160,
                      "description": "Search terms preserving the supplied conversation's entity, requested relation and applicability; do not add unprovided patient details."},
        },
    }
    capability = ToolCapability(
        tool_id=HEALTHBENCH_KNOWLEDGE_TOOL_ID, dataset_scope=("healthbench_professional",),
        action_schemas={"search": arguments}, input_schema=arguments,
        output_schema={"type": "object", "required": ["operation", "query", "database", "status", "evidence", "conversation_matches", "index_receipt"],
                       "properties": {"operation": {"const": "search"}, "query": {"type": "string"},
                                      "database": {"enum": ["conversation", "medical_references", "drug_labels"]},
                                      "evidence": {"type": "array"}, "conversation_matches": {"type": "array"},
                                      "status": {"type": "string"}, "index_receipt": {"type": ["object", "null"]}}},
        side_effect="none", timeout_seconds=20.0, version=HEALTHBENCH_KNOWLEDGE_VERSION,
    )
    registrations.append(ToolRegistration(HEALTHBENCH_KNOWLEDGE_TOOL_ID, _KnowledgeSearchBackend(), capability))
    return ToolRegistry(tuple(registrations))


def open_healthbench_knowledge_tool_registry(**kwargs):
    opened = open_healthbench_clinical_tool_registry(**kwargs)
    try:
        opened.registry = build_healthbench_knowledge_tool_registry(opened.registry)
    except BaseException:
        opened.close()
        raise
    return opened


class HealthBenchKnowledgeReactExecutionAdapter(HealthBenchClinicalReactExecutionAdapter):
    def __init__(self, *, knowledge_root: str | Path,
                 skillflow_source: str | Path = DEFAULT_SKILLFLOW_SOURCE, **kwargs):
        super().__init__(**kwargs)
        self.knowledge_root = Path(knowledge_root)
        self.skillflow_source = Path(skillflow_source)

    def _contract(self, request, observations):
        value = super()._contract(request, observations)
        store = _CURRENT_STORE.get()
        if store is not None:
            value += ("\nLocal retrieval database counts: " + json.dumps(store.counts())
                      + ". Conversation entries retain the original speaker and are not external medical evidence. "
                      "Use the unchanged full conversation to interpret entities and applicability. "
                      "Empty external indexes mean no indexed evidence, not that no evidence exists.")
        return value

    def _tool_action_error(self, *, request, action, observations):
        if action.resource_id != HEALTHBENCH_KNOWLEDGE_TOOL_ID:
            return super()._tool_action_error(request=request, action=action, observations=observations)
        # This local index changes after a real retrieval. A previously empty
        # query may therefore be retried, unlike immutable source API requests.
        store = _CURRENT_STORE.get()
        current = store.counts() if store is not None else None
        database = action.arguments.get("database")
        for observation in reversed(observations):
            prior = observation.get("executed_action")
            if not isinstance(prior, Mapping) or prior.get("resource_id") != action.resource_id or prior.get("arguments") != action.arguments:
                continue
            result = observation.get("result")
            previous_counts = result.get("database_counts") if isinstance(result, Mapping) else None
            if isinstance(previous_counts, Mapping) and current is not None:
                if previous_counts.get(database) != current.get(database):
                    return None
            return "duplicate_tool_request"
        return None

    async def execute(self, request: AgentRequest):
        store = HealthBenchKnowledgeStore(
            self.knowledge_root, parse_model_visible_conversation(request.problem),
            skillflow_source=self.skillflow_source,
        )
        token = _CURRENT_STORE.set(store)
        try:
            with (store.directory / "request_receipt.json").open("x", encoding="utf-8") as handle:
                json.dump({"request_id": request.request_id, "run_id": request.run_id,
                           "agent_id": request.agent.id, "model_id": request.model.model_id,
                           "graph_revision": request.graph_revision, "phase": request.phase.value}, handle)
            errors = _ingest_receipts(store, list(_routed_receipts(request)))
            response = await super().execute(request)
            if isinstance(response, AgentResponse):
                return replace(response, metadata={**response.metadata, "knowledge_index": {
                    "directory": str(store.directory), "counts": store.counts(),
                    "scope": "current_invocation_and_routed_tool_receipts",
                    "initial_index_error_types": errors,
                }})
            return response
        finally:
            _CURRENT_STORE.reset(token)
            store.close()
