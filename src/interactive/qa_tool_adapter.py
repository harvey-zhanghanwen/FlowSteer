"""SkillFlow public-retrieval backends for the unified ToolRegistry.

This module is a thin adapter over SkillFlow
``skillev.benchmarks.retrieval.RetrievalIndex``.  It preserves the upstream
read-only ``open -> search/read`` call chain and only projects public corpus
observations.  Benchmark task IDs, accepted answers, and evaluator state are
not accepted by either tool schema.
"""

from __future__ import annotations

import asyncio
from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor
from contextvars import ContextVar
from dataclasses import dataclass, field, replace
import inspect
import json
from pathlib import Path
import re
from tempfile import TemporaryDirectory
from typing import Callable, Mapping, Optional, Protocol, Sequence
import unicodedata

from .agent_runtime import AgentGateway, AgentRequest, GatewayResponse
from .react_execution import ReactExecutionError, ToolReactExecutionAdapter
from .task_dataset import (
    hotpotqa_answer_cardinality_constraint,
    hotpotqa_answer_type_constraint,
    hotpotqa_question_scope,
    qa_answer_cardinality_constraint,
    qa_answer_type_constraint,
    qa_question_scope,
)
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
QA_VERIFIED_ANSWER_LINEAGE_PROTOCOL = "qa_verified_answer_lineage_v2"
HOTPOTQA_SEMANTIC_LINEAGE_PROTOCOL = "hotpotqa_semantic_lineage_v2"
HOTPOTQA_ROLE_CONDITIONAL_PROTOCOL = (
    "hotpotqa_role_conditional_capabilities_v1"
)
_HOTPOTQA_STRUCTURED_REASONER_PROTOCOLS = frozenset(
    {
        "hotpotqa_verified_answer_slot_v1",
        HOTPOTQA_SEMANTIC_LINEAGE_PROTOCOL,
        HOTPOTQA_ROLE_CONDITIONAL_PROTOCOL,
    }
)
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
    "question's exact scope and answer slot. Resolve entity aliases and coreference "
    "from the supplied passages before composing the retrieval query, and retain "
    "that entity binding through every hop. Apply the original wh-word answer type: "
    "a Which-comparison returns the compared entity, not the numeric/date comparison "
    "value; a who-question returns the evidence-supported answer-bearing entity "
    "(which may be a person or organization), not a possessive attribute phrase. "
    "Represent retrieved facts as "
    "subject/entity, predicate/relation, object or attribute value, and qualifiers. "
    "Preserve the sentence's asserted semantic roles instead of placing the desired "
    "candidate into an unrelated proposition field. In a comparison proposition, "
    "the compared entity is normally the subject and its compared date, number, or "
    "attribute is object_or_attribute_value. Copy entity surfaces exactly from each "
    "proposition's evidence span; do not silently replace a pronoun, surname, or "
    "shortened mention with the longer question entity. Use an antecedent-bearing "
    "span or a separate identity proposition supported by the same read receipt. "
    "For quantitative comparisons, missing explicit scope-matching count or event "
    "evidence is unknown, not zero. Count only explicit distinct events in the "
    "original requested category and do not replace an aggregate scope with a "
    "narrower subtype. "
    "Bind candidate_answer to exactly one evidence_propositions item through "
    "answer_slot.proposition_index and answer_field; keep the entity-to-attribute "
    "binding explicit and show every bridge in the multi-hop chain. Return one "
    "minimal but complete evidence-aligned referential surface when "
    "answer_cardinality is single; do not return an alias list, appositive gloss, "
    "or the question's answer-type head noun. For a who-question whose evidence "
    "expresses the requested person through a possessive construction, exclude the "
    "possessive marker and possessed attribute, but retain the complete possessor "
    "entity mention immediately before the marker, including any title, honorific, "
    "or name suffix present in the evidence. If compared values "
    "are unexpectedly equal, recheck scope, "
    "both bindings, retrieved passages, and contract narrowing before calling a tie."
)
QA_VERIFIED_ANSWER_LINEAGE_GUIDANCE = (
    "Treat ReAct only as the execution schedule Thought -> Action(tool) -> "
    "Observation -> Thought -> Final, never as an Agent role. Preserve the "
    "question's exact semantic scope and answer slot. Bind the target entity, "
    "requested relation, and every answer-bearing proposition to successful "
    "qa-retrieval read receipts. Keep spelling variants, aliases, and canonical "
    "names explicit in the propositions; do not guess an entity identity or "
    "relation that is absent from retrieved evidence. Return a concise semantic "
    "answer and leave surface-only output formatting to the Format Agent."
)

_HOTPOTQA_SEMANTIC_STRUCTURE_ERROR_PREFIX = (
    "hotpotqa_semantic_artifact_invalid:"
)
_HOTPOTQA_SEMANTIC_EVIDENCE_ERROR_PREFIX = (
    "hotpotqa_semantic_evidence_provenance_invalid:"
)
_QA_SEMANTIC_STRUCTURE_ERROR_PREFIX = "qa_semantic_artifact_invalid:"
_QA_SEMANTIC_EVIDENCE_ERROR_PREFIX = (
    "qa_semantic_evidence_provenance_invalid:"
)
_QA_MISSING_EVIDENCE_ERROR = "qa_completion_requires_successful_read_evidence"
_KNOWLEDGE_BASE_COVERAGE_FAILURE = "knowledge_base_coverage_failure"

# PROJECT_NECESSARY_ADAPTATION: SkillFlow supplies the public search/read
# actions and bounded continuation, while unified_architecture_v2 requires a
# bounded factual-QA recovery policy.  The initial top-k is the existing
# TriviaQA search limit; each public retry broadens it without changing the
# upstream retrieval backend.
_FACTUAL_QA_RETRIEVAL_STRATEGIES = (
    "initial_retrieval",
    "spelling_normalization",
    "alias_expansion",
    "entity_disambiguation",
    "query_rewriting",
)
_FACTUAL_QA_SEARCH_LIMITS = (5, 10, 15, 20, 25)

_CALENDAR_MONTH_PATTERN = (
    r"(?:january|february|march|april|may|june|july|august|september|"
    r"october|november|december|jan|feb|mar|apr|jun|jul|aug|sep|sept|"
    r"oct|nov|dec)"
)

_QUESTION_ANCHOR_WH_WORDS = frozenset(
    {
        "what",
        "which",
        "who",
        "whom",
        "whose",
        "where",
        "when",
        "why",
        "how",
    }
)


def _strong_answer_surface_type(surface: str) -> str | None:
    """Classify only unambiguous answer surfaces without external knowledge.

    This deliberately leaves names and places unclassified: deciding whether
    an arbitrary proper noun is a person or location requires knowledge that
    the answer-free Retriever does not own.  Calendar and scalar surfaces are
    sufficiently explicit to reject a read that fills the wrong relation
    argument (for example, a birth date for a birthplace question).
    """

    normalized = " ".join(surface.casefold().split()).strip(" .,:;()[]")
    if normalized in {"yes", "no"}:
        return "yes_no"
    if re.fullmatch(r"(?:1[0-9]{3}|20[0-9]{2})", normalized):
        return "year_or_number"
    if re.search(rf"\b{_CALENDAR_MONTH_PATTERN}\b", normalized) and re.search(
        r"\b\d{1,4}\b", normalized
    ):
        return "date"
    if re.fullmatch(
        r"\d{1,4}[-/]\d{1,2}(?:[-/]\d{1,4})?",
        normalized,
    ):
        return "date"
    if re.fullmatch(
        r"(?:the\s+)?(?:\d{1,2}(?:st|nd|rd|th)|[a-z-]+)\s+century",
        normalized,
    ):
        return "date"
    if re.fullmatch(
        r"[+-]?(?:\d+(?:[,.]\d+)*|\.\d+)(?:\s*(?:%|percent))?",
        normalized,
    ):
        return "number"
    return None


def _answer_surface_type_issue(*, expected_type: str, surface: str) -> str | None:
    """Return a conservative question-slot/evidence-argument mismatch."""

    observed_type = _strong_answer_surface_type(surface)
    admitted_observed_types = {
        "date": {"date", "year_or_number"},
        "number": {"number", "year_or_number"},
        "yes_no": {"yes_no"},
    }
    if expected_type in admitted_observed_types:
        if observed_type not in admitted_observed_types[expected_type]:
            return (
                "Evidence Retriever evidence proposition does not supply the "
                f"question's requested {expected_type} relation argument"
            )
        return None
    if expected_type in {"location", "entity", "nationality"} and observed_type in {
        "date",
        "number",
        "year_or_number",
        "yes_no",
    }:
        return (
            "Evidence Retriever evidence proposition supplies a "
            f"{observed_type.replace('_', '/')} relation argument, not the "
            "question's requested "
            f"{expected_type} relation argument"
        )
    return None


def _normalized_retrieval_query(query: str) -> str:
    """Canonicalize only for duplicate-request admission, not retrieval."""

    normalized = unicodedata.normalize("NFKC", query).casefold()
    return " ".join(re.findall(r"\w+", normalized, flags=re.UNICODE))


@dataclass(frozen=True, slots=True)
class _RequiredEvidenceState:
    """Public SkillFlow search/read state used by the QA action mask."""

    required: bool
    search_queries: tuple[str, ...]
    normalized_search_queries: tuple[str, ...]
    search_top_ks: tuple[int, ...]
    search_attempt_count: int
    searched_passage_ids: tuple[str, ...]
    latest_search_passage_ids: tuple[str, ...]
    read_passage_ids: tuple[str, ...]
    dispatched_tool_calls: int
    latest_successful_operation: str | None
    semantic_repair_kind: str | None
    semantic_repair_error_code: str | None

    @property
    def unread_passage_ids(self) -> tuple[str, ...]:
        read = frozenset(self.read_passage_ids)
        return tuple(
            passage_id
            for passage_id in self.searched_passage_ids
            if passage_id not in read
        )

    @property
    def latest_unread_passage_ids(self) -> tuple[str, ...]:
        read = frozenset(self.read_passage_ids)
        return tuple(
            passage_id
            for passage_id in self.latest_search_passage_ids
            if passage_id not in read
        )

    @property
    def successful_read_count(self) -> int:
        return len(self.read_passage_ids)


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
        self._semantic_reasoner_question: ContextVar[str | None] = ContextVar(
            f"qa_semantic_reasoner_question_{id(self)}",
            default=None,
        )
        self._semantic_reasoner_protocol: ContextVar[str | None] = ContextVar(
            f"qa_semantic_reasoner_protocol_{id(self)}",
            default=None,
        )
        self._semantic_evidence_retriever_question: ContextVar[
            str | None
        ] = ContextVar(
            f"qa_semantic_evidence_retriever_question_{id(self)}",
            default=None,
        )
        self._semantic_evidence_retriever_protocol: ContextVar[
            str | None
        ] = ContextVar(
            f"qa_semantic_evidence_retriever_protocol_{id(self)}",
            default=None,
        )
        self._semantic_upstream_tool_receipts: ContextVar[
            tuple[Mapping[str, object], ...]
        ] = ContextVar(
            f"qa_semantic_upstream_tool_receipts_{id(self)}",
            default=(),
        )

    @staticmethod
    def _role_conditional_upstream_candidate(
        request: AgentRequest,
    ) -> tuple[bool, Optional[str]]:
        """Return an exact routed candidate only for an unmasked Output call."""

        if (
            request.semantic_protocol != HOTPOTQA_ROLE_CONDITIONAL_PROTOCOL
            or not request.is_output_agent
            or (request.agent.role_family or "").casefold() != "output"
            or request.communication_condition.value != "normal"
        ):
            return False, None
        from .agent_workflow_env import AgentWorkflowEnv

        candidates: list[str] = []
        for message in request.upstream:
            candidate, issue = AgentWorkflowEnv._semantic_candidate_from_artifact(
                message.artifact
            )
            if issue is not None:
                return True, None
            if candidate is not None:
                candidates.append(candidate)
        if not candidates:
            return False, None
        candidate = candidates[0]
        if any(item != candidate for item in candidates):
            return True, None
        return True, candidate

    def _unified_factual_protocol(self, request: AgentRequest) -> bool:
        return (
            self._task_type == "factual_qa"
            and request.semantic_protocol == QA_VERIFIED_ANSWER_LINEAGE_PROTOCOL
        )

    @staticmethod
    def _direct_upstream_tool_receipts(
        request: AgentRequest,
    ) -> tuple[Mapping[str, object], ...]:
        """Return direct predecessor receipts without changing own Tool budget."""

        return tuple(
            receipt
            for message in request.upstream
            for receipt in message.tool_receipts
            if isinstance(receipt, Mapping)
        )

    @staticmethod
    def _successful_read_receipt(receipt: Mapping[str, object]) -> bool:
        if receipt.get("tool_id") != QA_RETRIEVAL_TOOL_ID:
            return False
        request = receipt.get("request")
        result = receipt.get("result")
        if (
            not isinstance(request, Mapping)
            or request.get("action") != "read"
            or not isinstance(result, Mapping)
            or receipt.get("error_type") is not None
        ):
            return False
        value = result.get("value")
        return (
            isinstance(value, Mapping)
            and value.get("operation") == "read"
            and isinstance(value.get("passage"), Mapping)
            and isinstance(value["passage"].get("text"), str)
            and bool(value["passage"]["text"].strip())
        )

    def _hotpot_tool_plan_exhausted(
        self,
        request: AgentRequest,
        state: _RequiredEvidenceState,
    ) -> bool:
        """Return a typed bounded-retrieval diagnosis, never a task oracle."""

        if (
            request.semantic_protocol not in _HOTPOTQA_STRUCTURED_REASONER_PROTOCOLS
            or (request.agent.role_family or "").casefold() != "reasoner"
            or state.semantic_repair_kind != "evidence"
        ):
            return False
        remaining_tool_calls = max(
            0,
            self._max_tool_calls - state.dispatched_tool_calls,
        )
        if (
            state.latest_successful_operation == "search"
            and state.latest_unread_passage_ids
            and remaining_tool_calls >= 1
        ):
            return False
        return remaining_tool_calls < 2

    @staticmethod
    def _factual_retrieval_strategy(search_attempt_count: int) -> str:
        index = min(
            max(search_attempt_count, 0),
            len(_FACTUAL_QA_RETRIEVAL_STRATEGIES) - 1,
        )
        return _FACTUAL_QA_RETRIEVAL_STRATEGIES[index]

    @staticmethod
    def _factual_search_limit(search_attempt_count: int) -> int:
        index = min(
            max(search_attempt_count, 0),
            len(_FACTUAL_QA_SEARCH_LIMITS) - 1,
        )
        return _FACTUAL_QA_SEARCH_LIMITS[index]

    @staticmethod
    def _semantic_rejection_kind(public_error_code: object) -> str | None:
        """Classify public completion feedback without inspecting hidden labels."""

        if not isinstance(public_error_code, str):
            return None
        if (
            public_error_code == _QA_MISSING_EVIDENCE_ERROR
            or public_error_code.startswith(
                _HOTPOTQA_SEMANTIC_EVIDENCE_ERROR_PREFIX
            )
            or public_error_code.startswith(
                _QA_SEMANTIC_EVIDENCE_ERROR_PREFIX
            )
        ):
            return "evidence"
        if public_error_code.startswith(
            _HOTPOTQA_SEMANTIC_STRUCTURE_ERROR_PREFIX
        ) or public_error_code.startswith(_QA_SEMANTIC_STRUCTURE_ERROR_PREFIX):
            return "structure"
        if public_error_code == _KNOWLEDGE_BASE_COVERAGE_FAILURE:
            return "coverage"
        return None

    @staticmethod
    def _semantic_repair_instruction(repair_kind: str) -> str:
        """Return the public SkillFlow continuation instruction for one fault."""

        if repair_kind == "structure":
            return (
                "Preserve all successful qa-retrieval read evidence and every "
                "semantic field not implicated by this public_error_code. Repair "
                "only the diagnosed structured semantic artifact fields, then "
                "emit a complete action; do not add a search or read."
            )
        if repair_kind == "evidence":
            return (
                "Preserve the current semantic work and use the admitted "
                "qa-retrieval search/read continuation to obtain the missing "
                "evidence or provenance before completing again."
            )
        if repair_kind == "coverage":
            return (
                "The bounded retrieval strategies and Tool budget did not produce "
                "evidence that binds the target entity and relation. Do not guess "
                "or fabricate an answer or evidence."
            )
        raise ValueError(f"unsupported semantic repair kind {repair_kind!r}")

    @classmethod
    def _model_visible_observations(
        cls,
        observations: list[Mapping[str, object]],
    ) -> list[dict[str, object]]:
        """Attach the public continuation repair to semantic rejections.

        SkillFlow carries Action--Observation feedback into the next model
        turn.  Keep the sampled invalid completion only in the trajectory and
        expose a diagnosis that says whether the existing read evidence must
        be augmented or only the structured completion must be repaired.
        """

        visible = ToolReactExecutionAdapter._model_visible_observations(
            observations
        )
        for observation in visible:
            public_error_code = observation.get("public_error_code")
            answer_field_mismatch = (
                re.search(
                    r"Reasoner answer_slot\.answer_field selects '([^']+)'"
                    r".*candidate_answer exactly matches the selected "
                    r"proposition field '([^']+)'",
                    public_error_code,
                )
                if isinstance(public_error_code, str)
                else None
            )
            if answer_field_mismatch is not None:
                selected_field, evidence_field = answer_field_mismatch.groups()
                observation["repair_instruction"] = (
                    "Repair only the diagnosed structured semantic artifact "
                    "fields. Preserve the successful read receipts, question_scope, "
                    "evidence propositions, candidate_answer, and every valid "
                    "answer-slot field. Change only answer_slot.answer_field "
                    f"from {selected_field!r} to {evidence_field!r}, the unique "
                    "proposition field already containing candidate_answer; do "
                    "not add a search or read, and do not replace the candidate."
                )
                continue
            if isinstance(public_error_code, str) and (
                "entity_identity.evidence_surface is not supported by the "
                "cited passage title identity chain"
            ) in public_error_code:
                observation["repair_instruction"] = (
                    "Preserve the same successful read receipt, passage_id, passage "
                    "title, target_relation, evidence_span, and proposition. Repair "
                    "only entity_identity: keep question_surface on the original "
                    "question entity and copy its coreferential evidence_surface from "
                    "both that same passage title and exact evidence_span. Do not "
                    "change the proposition. Emit a complete action; do not search "
                    "or read again."
                )
                continue
            if isinstance(public_error_code, str) and (
                "Evidence Retriever evidence_span has no typography-canonical "
                "lexical match"
            ) in public_error_code:
                observation["repair_instruction"] = (
                    "Preserve the cited passage_id and all successful Tool receipts. "
                    "Copy one contiguous exact evidence_span from that same public "
                    "qa-retrieval read receipt, allowing only typography/whitespace "
                    "canonicalization; do not paraphrase, concatenate spans, search, "
                    "or read again."
                )
                continue
            if public_error_code == "qa_retrieval_duplicate_normalized_query":
                observation["repair_instruction"] = (
                    "Preserve all successful Tool receipts and issue a "
                    "semantically distinct entity-and-relation query using the "
                    "current retrieval strategy; do not repeat a prior query "
                    "after Unicode normalization and case folding."
                )
                continue
            repair_kind = cls._semantic_rejection_kind(public_error_code)
            if repair_kind is not None:
                observation["repair_instruction"] = (
                    cls._semantic_repair_instruction(repair_kind)
                )

        # SkillFlow persists every sampled Action--Observation turn in the
        # trajectory, but the next model input need not replay an unbounded run
        # of identical invalid-action observations. Preserve every Tool result
        # and every change of diagnosis while collapsing only consecutive
        # duplicate public errors.
        compacted: list[dict[str, object]] = []
        last_invalid_key: tuple[str, str] | None = None
        for observation in visible:
            status = observation.get("observation_status")
            public_error_code = observation.get("public_error_code")
            if (
                status in {"parse_error", "schema_invalid"}
                and isinstance(public_error_code, str)
            ):
                invalid_key = (str(status), public_error_code)
                if invalid_key == last_invalid_key:
                    prior = compacted[-1]
                    repeat_count = prior.get("repeat_count", 1)
                    prior["repeat_count"] = (
                        repeat_count + 1
                        if isinstance(repeat_count, int)
                        and not isinstance(repeat_count, bool)
                        else 2
                    )
                    continue
                last_invalid_key = invalid_key
            else:
                last_invalid_key = None
            compacted.append(observation)
        return compacted

    def _required_evidence_state(
        self,
        request: AgentRequest,
        observations: list[Mapping[str, object]],
    ) -> _RequiredEvidenceState:
        required = (
            self._completion_policy == "required_evidence"
            and self._max_tool_calls > 0
            and QA_RETRIEVAL_TOOL_ID in request.agent.allowed_tools
        )
        search_queries: list[str] = []
        normalized_search_queries: list[str] = []
        search_top_ks: list[int] = []
        searched_passage_ids: list[str] = []
        latest_search_passage_ids: list[str] = []
        read_passage_ids: list[str] = []
        dispatched_tool_calls = 0
        latest_successful_operation: str | None = None
        latest_successful_read_index = -1
        latest_semantic_rejection_index = -1
        latest_semantic_rejection_kind: str | None = None
        latest_semantic_rejection_code: str | None = None
        for observation_index, observation in enumerate(observations):
            status = observation.get("observation_status")
            public_error_code = observation.get("public_error_code")
            semantic_rejection_kind = self._semantic_rejection_kind(
                public_error_code
            )
            if status == "schema_invalid" and semantic_rejection_kind is not None:
                latest_semantic_rejection_index = observation_index
                latest_semantic_rejection_kind = semantic_rejection_kind
                assert isinstance(public_error_code, str)
                latest_semantic_rejection_code = public_error_code
            if status not in {"success", "tool_error"}:
                continue
            executed_action = observation.get("executed_action")
            if isinstance(executed_action, Mapping):
                if (
                    executed_action.get("kind") == "tool"
                    and executed_action.get("resource_id") == QA_RETRIEVAL_TOOL_ID
                ):
                    dispatched_tool_calls += 1
                    if executed_action.get("name") == "search":
                        arguments = executed_action.get("arguments")
                        if isinstance(arguments, Mapping):
                            query = arguments.get("query")
                            top_k = arguments.get("limit")
                            if isinstance(query, str) and query.strip():
                                query = query.strip()
                                search_queries.append(query)
                                normalized_search_queries.append(
                                    _normalized_retrieval_query(query)
                                )
                                if type(top_k) is int and top_k > 0:
                                    search_top_ks.append(top_k)
            elif status == "success":
                # Unit fixtures and restored legacy public observations may
                # omit executed_action.  A successful retrieval result still
                # represents exactly one dispatched SkillFlow Tool action.
                dispatched_tool_calls += 1
            if status != "success":
                continue
            result = observation.get("result")
            if not isinstance(result, Mapping):
                continue
            if result.get("operation") == "search":
                latest_successful_operation = "search"
                latest_search_passage_ids = []
                # Legacy fixtures may omit executed_action but retain the
                # public SkillFlow query/top-k result.  Rehydrate the bounded
                # attempt state from those public fields only.
                if not isinstance(executed_action, Mapping):
                    query = result.get("query")
                    top_k = result.get("top_k")
                    if isinstance(query, str) and query.strip():
                        query = query.strip()
                        search_queries.append(query)
                        normalized_search_queries.append(
                            _normalized_retrieval_query(query)
                        )
                        if type(top_k) is int and top_k > 0:
                            search_top_ks.append(top_k)
                raw_ids = result.get("passage_ids")
                if isinstance(raw_ids, list):
                    for value in raw_ids:
                        if not isinstance(value, str) or not value.strip():
                            continue
                        passage_id = value.strip()
                        if passage_id not in searched_passage_ids:
                            searched_passage_ids.append(passage_id)
                        if passage_id not in latest_search_passage_ids:
                            latest_search_passage_ids.append(passage_id)
            elif (
                result.get("operation") == "read"
                and isinstance(result.get("passage"), Mapping)
                and isinstance(result["passage"].get("text"), str)
                and bool(result["passage"]["text"].strip())
            ):
                latest_successful_operation = "read"
                latest_successful_read_index = observation_index
                raw_passage_id = result.get("passage_id")
                if not isinstance(raw_passage_id, str):
                    raw_passage_id = result["passage"].get("passage_id")
                if not isinstance(raw_passage_id, str) and isinstance(
                    executed_action, Mapping
                ):
                    arguments = executed_action.get("arguments")
                    if isinstance(arguments, Mapping):
                        raw_passage_id = arguments.get("passage_id")
                if isinstance(raw_passage_id, str) and raw_passage_id.strip():
                    passage_id = raw_passage_id.strip()
                    if passage_id not in read_passage_ids:
                        read_passage_ids.append(passage_id)
        return _RequiredEvidenceState(
            required=required,
            search_queries=tuple(search_queries),
            normalized_search_queries=tuple(normalized_search_queries),
            search_top_ks=tuple(search_top_ks),
            search_attempt_count=len(search_queries),
            searched_passage_ids=tuple(searched_passage_ids),
            latest_search_passage_ids=tuple(latest_search_passage_ids),
            read_passage_ids=tuple(read_passage_ids),
            dispatched_tool_calls=dispatched_tool_calls,
            latest_successful_operation=latest_successful_operation,
            semantic_repair_kind=(
                latest_semantic_rejection_kind
                if latest_semantic_rejection_index > latest_successful_read_index
                else None
            ),
            semantic_repair_error_code=(
                latest_semantic_rejection_code
                if latest_semantic_rejection_index > latest_successful_read_index
                else None
            ),
        )

    def _unified_factual_action_domain(
        self,
        state: _RequiredEvidenceState,
    ) -> tuple[frozenset[tuple[str, str]], bool]:
        """Apply bounded factual-QA retrieval recovery to public state."""

        remaining_tool_calls = max(
            0,
            self._max_tool_calls - state.dispatched_tool_calls,
        )
        strategies_exhausted = (
            state.search_attempt_count >= len(_FACTUAL_QA_RETRIEVAL_STRATEGIES)
        )

        # A structured semantic error is repaired on preserved evidence.  A
        # provenance/entity/relation mismatch instead advances the public
        # search strategy; it never admits a guessed completion.
        if state.semantic_repair_kind == "structure":
            return frozenset(), state.successful_read_count > 0
        if state.semantic_repair_kind == "coverage":
            return frozenset(), False
        if state.semantic_repair_kind == "evidence":
            if (
                state.latest_successful_operation == "search"
                and state.latest_unread_passage_ids
                and remaining_tool_calls >= 1
            ):
                return frozenset({(QA_RETRIEVAL_TOOL_ID, "read")}), False
            if strategies_exhausted or remaining_tool_calls < 2:
                return frozenset(), False
            return frozenset({(QA_RETRIEVAL_TOOL_ID, "search")}), False

        if state.successful_read_count > 0:
            return frozenset(), True
        if state.latest_unread_passage_ids and remaining_tool_calls >= 1:
            return frozenset({(QA_RETRIEVAL_TOOL_ID, "read")}), False
        if strategies_exhausted or remaining_tool_calls < 2:
            return frozenset(), False
        return frozenset({(QA_RETRIEVAL_TOOL_ID, "search")}), False

    def _state_conditioned_action_domain(
        self,
        request: AgentRequest,
        observations: list[Mapping[str, object]],
    ) -> tuple[Optional[frozenset[tuple[str, str]]], bool]:
        """Expose a bounded multi-hop search/read continuation domain."""

        state = self._required_evidence_state(request, observations)
        if not state.required:
            return super()._state_conditioned_action_domain(request, observations)
        upstream_completion_admitted = any(
            self._successful_read_receipt(receipt)
            for receipt in self._direct_upstream_tool_receipts(request)
        )

        def admitted(
            tool_actions: Optional[frozenset[tuple[str, str]]],
            completion: bool,
        ) -> tuple[Optional[frozenset[tuple[str, str]]], bool]:
            # A direct Retriever predecessor may satisfy semantic provenance,
            # but its receipts remain outside this semantic Agent's Tool budget.
            # Completion validation below still checks every cited span.  Once
            # that exact upstream evidence has produced an evidence/provenance
            # rejection, preserve it but revoke completion until this Agent
            # obtains a new successful read through the public SkillFlow
            # Action--Observation continuation.  Otherwise one irrelevant
            # predecessor read permanently masks the search/read recovery
            # domain and causes repeated invalid completion actions.
            upstream_can_complete = (
                upstream_completion_admitted
                and state.semantic_repair_kind not in {"evidence", "coverage"}
            )
            return tool_actions, completion or upstream_can_complete

        if self._unified_factual_protocol(request):
            tool_actions, completion = self._unified_factual_action_domain(state)
            return admitted(tool_actions, completion)

        remaining_tool_calls = max(
            0,
            self._max_tool_calls - state.dispatched_tool_calls,
        )
        minimum_reads = (
            2
            if self._task_type == "multi_hop_qa"
            and request.semantic_protocol in _HOTPOTQA_STRUCTURED_REASONER_PROTOCOLS
            else 1
        )
        hotpot_multi_hop = (
            self._task_type == "multi_hop_qa"
            and request.semantic_protocol in _HOTPOTQA_STRUCTURED_REASONER_PROTOCOLS
        )

        # SkillFlow's public Action--Observation continuation distinguishes a
        # missing evidence/provenance diagnosis from a structured completion
        # diagnosis.  The latter must repair on the already-read evidence;
        # blindly retrieving again cannot fix answer-slot or schema binding.
        if state.semantic_repair_kind == "structure":
            return admitted(frozenset(), state.successful_read_count > 0)
        if state.semantic_repair_kind == "evidence":
            if (
                state.latest_successful_operation == "search"
                and state.latest_unread_passage_ids
                and remaining_tool_calls >= 1
            ):
                return admitted(
                    frozenset({(QA_RETRIEVAL_TOOL_ID, "read")}), False
                )
            if remaining_tool_calls >= 2:
                return admitted(
                    frozenset({(QA_RETRIEVAL_TOOL_ID, "search")}), False
                )
            return admitted(frozenset(), state.successful_read_count > 0)

        if (
            state.successful_read_count >= minimum_reads
            or remaining_tool_calls == 0
            and state.successful_read_count > 0
        ):
            return admitted(frozenset(), True)

        # HotpotQA multi-hop retrieval uses the newest read Observation to
        # formulate the next missing-hop search.  This preserves SkillFlow's
        # public Action--Observation continuation while avoiding two blind
        # reads from one initial query.
        if hotpot_multi_hop and state.successful_read_count > 0:
            if (
                state.latest_successful_operation == "search"
                and state.latest_unread_passage_ids
            ):
                return admitted(
                    frozenset({(QA_RETRIEVAL_TOOL_ID, "read")}), False
                )
            if remaining_tool_calls >= 2:
                return admitted(
                    frozenset({(QA_RETRIEVAL_TOOL_ID, "search")}), False
                )
            return admitted(frozenset(), True)

        if state.latest_unread_passage_ids:
            action_name = "read"
        elif remaining_tool_calls >= 2 or state.successful_read_count == 0:
            action_name = "search"
        else:
            # One Tool call cannot complete a new search->read transition.
            # Preserve the successfully read evidence and admit completion.
            return admitted(frozenset(), state.successful_read_count > 0)
        return admitted(
            frozenset({(QA_RETRIEVAL_TOOL_ID, action_name)}), False
        )

    def _completion_arguments_schema(
        self,
        request: AgentRequest,
    ) -> Mapping[str, object]:
        semantic_protocol = request.semantic_protocol
        semantic_role = (request.agent.role_family or "").casefold()
        candidate_seen, routed_candidate = (
            self._role_conditional_upstream_candidate(request)
        )
        if candidate_seen and routed_candidate is not None:
            return {
                "type": "object",
                "required": ["value"],
                "properties": {
                    "value": {
                        "const": f"<answer>{routed_candidate}</answer>",
                        "description": (
                            "Copy the single agreeing routed semantic candidate "
                            "character-for-character into the terminal answer "
                            "wrapper; do not reselect or rewrite it."
                        ),
                    }
                },
                "additionalProperties": False,
            }
        if (
            semantic_role == "evidence_retriever"
            and semantic_protocol == HOTPOTQA_ROLE_CONDITIONAL_PROTOCOL
        ):
            # SkillFlow's retrieval environment owns public search/read
            # observations.  In the role-conditional AgentGraph, the selected
            # Retriever therefore exports only receipt-grounded evidence;
            # predicate--argument and answer-slot alignment remain downstream
            # semantic-producer responsibilities.
            return {
                "type": "object",
                "required": ["value"],
                "properties": {
                    "value": {
                        "type": "array",
                        "minItems": 1,
                        "items": {
                            "type": "object",
                            "required": ["passage_id", "evidence_span"],
                            "properties": {
                                "passage_id": {
                                    "type": "string",
                                    "minLength": 1,
                                },
                                "evidence_span": {
                                    "type": "string",
                                    "minLength": 1,
                                },
                            },
                            "additionalProperties": False,
                        },
                    }
                },
                "additionalProperties": False,
            }
        if semantic_role == "evidence_retriever" and semantic_protocol in {
            *_HOTPOTQA_STRUCTURED_REASONER_PROTOCOLS,
            QA_VERIFIED_ANSWER_LINEAGE_PROTOCOL,
        }:
            non_empty_text = {"type": "string", "minLength": 1}
            question_scope = (
                hotpotqa_question_scope(request.problem)
                if semantic_protocol in _HOTPOTQA_STRUCTURED_REASONER_PROTOCOLS
                else qa_question_scope(request.problem)
            )
            answer_type = (
                hotpotqa_answer_type_constraint(request.problem)
                if semantic_protocol in _HOTPOTQA_STRUCTURED_REASONER_PROTOCOLS
                else qa_answer_type_constraint(request.problem)
            )
            # PROJECT_NECESSARY_ADAPTATION: SkillFlow constrains every public
            # StructuredAction and read receipt, while the shared AgentGraph
            # recovery branch needs an answer-free intermediate artifact.  The
            # Retriever owns evidence provenance only; semantic answer
            # selection remains exclusively with the Reasoner.
            return {
                "type": "object",
                "required": ["value"],
                "properties": {
                    "value": {
                        "type": "object",
                        "required": [
                            "question_scope",
                            "entity_identity",
                            "target_relation",
                            "answer_type_constraint",
                            "evidence_proposition",
                            "evidence_span",
                            "passage_id",
                        ],
                        "properties": {
                            "question_scope": {
                                "const": question_scope,
                                "description": (
                                    "Copy the original question exactly without "
                                    "narrowing its semantic scope."
                                ),
                            },
                            "entity_identity": {
                                "type": "object",
                                "required": [
                                    "question_surface",
                                    "evidence_surface",
                                ],
                                "properties": {
                                    "question_surface": {
                                        **non_empty_text,
                                        "description": (
                                            "The concise question-side entity or "
                                            "event anchor copied from the original "
                                            "question. It must not be a wh-word or "
                                            "wh-phrase, the whole question, or a "
                                            "candidate/final answer."
                                            + (
                                                " For factual QA, an exact cited "
                                                "passage-title binding may normalize "
                                                "only one leading honorific and must "
                                                "come from the same successful read "
                                                "receipt."
                                                if semantic_protocol
                                                == QA_VERIFIED_ANSWER_LINEAGE_PROTOCOL
                                                else ""
                                            )
                                        ),
                                    },
                                    "evidence_surface": {
                                        **non_empty_text,
                                        "description": (
                                            "The coreferential surface of the same "
                                            "question-side entity or event anchor, "
                                            "copied from the receipt-grounded exact "
                                            "evidence_span, not the whole span."
                                            + (
                                                " For factual QA, when a distinct "
                                                "question_surface is exactly bound to "
                                                "the cited passage title, this same-"
                                                "entity surface must occur in both "
                                                "that title and the exact evidence_span."
                                                if semantic_protocol
                                                == QA_VERIFIED_ANSWER_LINEAGE_PROTOCOL
                                                else ""
                                            )
                                        ),
                                    },
                                },
                                "additionalProperties": False,
                            },
                            "target_relation": {
                                **non_empty_text,
                                "description": (
                                    "Copy the exact predicate surface from "
                                    "evidence_span (for example, `was born in`), "
                                    "not an abstract relation label."
                                ),
                            },
                            "answer_type_constraint": {
                                "const": answer_type,
                                "description": (
                                    "The question-only answer type returned by "
                                    "qa_answer_type_constraint. When the anchor "
                                    "binds exactly one binary proposition argument, "
                                    "it checks the other argument without selecting "
                                    "an answer; answer-slot binding belongs to the "
                                    "Reasoner."
                                ),
                            },
                            "evidence_proposition": {
                                "type": "object",
                                "required": [
                                    "subject",
                                    "predicate",
                                    "object_or_attribute_value",
                                ],
                                "properties": {
                                    "subject": {
                                        **non_empty_text,
                                        "description": (
                                            "Exact subject/entity surface copied "
                                            "from evidence_span."
                                        ),
                                    },
                                    "predicate": {
                                        **non_empty_text,
                                        "description": (
                                            "Exact predicate surface copied from "
                                            "evidence_span; it must equal "
                                            "target_relation."
                                        ),
                                    },
                                    "object_or_attribute_value": {
                                        **non_empty_text,
                                        "description": (
                                            "Exact object or attribute-value "
                                            "surface copied from evidence_span. "
                                            "Together with subject and predicate, "
                                            "this records evidence provenance; it "
                                            "does not select a candidate answer."
                                        ),
                                    },
                                },
                                "additionalProperties": False,
                            },
                            "evidence_span": {
                                **non_empty_text,
                                "description": (
                                    "An exact supporting span from the cited read "
                                    "receipt."
                                ),
                            },
                            "passage_id": dict(non_empty_text),
                        },
                        "additionalProperties": False,
                    }
                },
                "additionalProperties": False,
            }
        if semantic_role != "reasoner" or (
            semantic_protocol
            not in {
                *_HOTPOTQA_STRUCTURED_REASONER_PROTOCOLS,
                QA_VERIFIED_ANSWER_LINEAGE_PROTOCOL,
            }
        ):
            return super()._completion_arguments_schema(request)
        unified_factual = self._unified_factual_protocol(request)
        minimum_reasoning_items = 1 if unified_factual else 2
        non_empty_text = {"type": "string", "minLength": 1}
        non_empty_text_list = {
            "type": "array",
            "minItems": 1,
            "items": dict(non_empty_text),
        }
        qualifier_list = {
            "type": "array",
            "items": dict(non_empty_text),
        }
        if semantic_protocol in _HOTPOTQA_STRUCTURED_REASONER_PROTOCOLS:
            entity_surface_description = (
                "When this field supplies an entity answer, copy one minimal but "
                "complete evidence-aligned referential surface. Do not truncate a "
                "title, honorific, or name suffix that belongs to the source entity "
                "mention. For a possessive construction, retain the complete possessor "
                "mention before the possessive marker and exclude the marker plus the "
                "possessed attribute."
            )
        else:
            entity_surface_description = (
                "When this field supplies an entity answer, use one concise "
                "evidence-grounded entity surface. Any spelling variant, alias, or "
                "canonical-name choice must be supported by an explicit identity "
                "binding in the evidence propositions."
            )
        answer_slot_schema = {
            "type": "object",
            "required": [
                "answer_type",
                "answer_cardinality",
                "qualifiers",
                "proposition_index",
                "answer_field",
            ],
            "properties": {
                "answer_type": {
                    "const": (
                        hotpotqa_answer_type_constraint(request.problem)
                        if semantic_protocol
                        in _HOTPOTQA_STRUCTURED_REASONER_PROTOCOLS
                        else qa_answer_type_constraint(request.problem)
                    ),
                    "description": "The answer type requested by the original question.",
                },
                "answer_cardinality": {
                    "const": (
                        hotpotqa_answer_cardinality_constraint(request.problem)
                        if semantic_protocol
                        in _HOTPOTQA_STRUCTURED_REASONER_PROTOCOLS
                        else qa_answer_cardinality_constraint(request.problem)
                    ),
                    "description": (
                        "Whether the original question requests one answer value or "
                        "multiple answer values."
                    ),
                },
                "qualifiers": dict(qualifier_list),
                "proposition_index": {
                    "type": "integer",
                    "minimum": 0,
                    "description": (
                        "Zero-based index of the evidence proposition whose selected "
                        "field supplies candidate_answer."
                    ),
                },
                "answer_field": {
                    "type": "string",
                    "enum": ["subject", "object_or_attribute_value"],
                    "description": (
                        "The selected proposition field copied as candidate_answer. "
                        "For an entity comparison, select the winning entity from "
                        "the subject field while its compared date, number, or "
                        "attribute remains object_or_attribute_value."
                    ),
                },
            },
            "additionalProperties": False,
        }

        proposition_schema = {
            "type": "object",
            "required": [
                "subject",
                "relation",
                "object_or_attribute_value",
                "qualifiers",
                "evidence_span",
            ],
            "properties": {
                "subject": {
                    **non_empty_text,
                    "description": entity_surface_description,
                },
                "relation": {
                    **non_empty_text,
                    "description": (
                        "The predicate asserted by the evidence sentence between "
                        "subject and object_or_attribute_value."
                    ),
                },
                "object_or_attribute_value": {
                    **non_empty_text,
                    "description": (
                        entity_surface_description
                        + " Preserve the value attributed to the subject; do not "
                        "repeat the subject merely to make it candidate_answer."
                    ),
                },
                "qualifiers": dict(qualifier_list),
                "evidence_span": {
                    **non_empty_text,
                    "description": "An exact supporting span from a read passage.",
                },
            },
            "additionalProperties": False,
        }
        # NECESSARY_ADAPTATION: SkillFlow constrains every StructuredAction at
        # its provider boundary.  The shared QA protocol additionally gives
        # the semantic Reasoner ownership of one exact six-field artifact. Nest that
        # artifact under completion arguments.value so structured serving
        # cannot confuse semantic fields with action-envelope fields.
        return {
            "type": "object",
            "required": ["value"],
            "properties": {
                "value": {
                    "type": "object",
                    "required": [
                        "question_scope",
                        "answer_slot",
                        "evidence_propositions",
                        "multi_hop_chain",
                        "candidate_answer",
                        "evidence",
                    ],
                    "properties": {
                        "question_scope": {
                            "const": (
                                hotpotqa_question_scope(request.problem)
                                if semantic_protocol
                                in _HOTPOTQA_STRUCTURED_REASONER_PROTOCOLS
                                else qa_question_scope(request.problem)
                            ),
                            "description": (
                                "Copy the original question exactly; do not narrow "
                                "or add qualifiers."
                            ),
                        },
                        "answer_slot": answer_slot_schema,
                        "evidence_propositions": {
                            "type": "array",
                            "minItems": minimum_reasoning_items,
                            "description": (
                                "Explicit answer-bearing and supporting propositions. "
                                "answer_slot.proposition_index selects the "
                                "answer-bearing proposition."
                            ),
                            "items": proposition_schema,
                        },
                        "multi_hop_chain": {
                            "type": "array",
                            "minItems": minimum_reasoning_items,
                            "items": dict(non_empty_text),
                        },
                        "candidate_answer": {
                            **non_empty_text,
                            "description": (
                                "Copy the selected proposition field exactly. For an "
                                "entity answer, it must be the minimal but complete "
                                "evidence-aligned referential surface described by "
                                "that field, not a strict subspan of the source entity "
                                "mention."
                            ),
                        },
                        "evidence": dict(non_empty_text_list),
                    },
                    "additionalProperties": False,
                }
            },
            "additionalProperties": False,
        }

    def _state_conditioned_response_schema(
        self,
        request: AgentRequest,
        observations: list[Mapping[str, object]],
    ) -> dict[str, object] | None:
        """Constrain read to unread opaque IDs returned by public search."""

        schema = super()._state_conditioned_response_schema(request, observations)
        if schema is None:
            return None
        # The generic builder shallow-copies a normalized Tool schema. Copy
        # this request-scoped schema before narrowing it so later Agents retain
        # the published SkillFlow capability unchanged.
        schema = deepcopy(schema)
        properties = schema.get("properties")
        if not isinstance(properties, dict):
            return schema
        name = properties.get("name")
        if not isinstance(name, Mapping):
            return schema
        action_name = name.get("const")
        arguments = properties.get("arguments")
        if not isinstance(arguments, dict):
            return schema
        argument_properties = arguments.get("properties")
        if not isinstance(argument_properties, dict):
            return schema
        if (
            action_name == "search"
            and request.semantic_protocol in _HOTPOTQA_STRUCTURED_REASONER_PROTOCOLS
        ):
            argument_properties["limit"] = {"const": 10}
            return schema
        if action_name == "search" and self._unified_factual_protocol(request):
            state = self._required_evidence_state(request, observations)
            strategy = self._factual_retrieval_strategy(
                state.search_attempt_count
            )
            argument_properties["query"] = {
                "type": "string",
                "minLength": 1,
                "description": (
                    "A new focused entity-and-relation query for retrieval "
                    f"strategy {strategy}; it must not repeat any prior query "
                    "after Unicode normalization and case folding."
                ),
            }
            argument_properties["limit"] = {
                "const": self._factual_search_limit(
                    state.search_attempt_count
                )
            }
            return schema
        if action_name != "read":
            return schema
        state = self._required_evidence_state(request, observations)
        if not state.latest_unread_passage_ids:
            return schema
        argument_properties["passage_id"] = {
            "type": "string",
            "enum": list(state.latest_unread_passage_ids),
        }
        return schema

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
        semantic_reasoner_protocol = (
            request.semantic_protocol
            if request.semantic_protocol
            in {
                *_HOTPOTQA_STRUCTURED_REASONER_PROTOCOLS,
                QA_VERIFIED_ANSWER_LINEAGE_PROTOCOL,
            }
            and (request.agent.role_family or "").casefold() == "reasoner"
            else None
        )
        semantic_evidence_retriever_protocol = (
            request.semantic_protocol
            if request.semantic_protocol
            in {
                *_HOTPOTQA_STRUCTURED_REASONER_PROTOCOLS,
                QA_VERIFIED_ANSWER_LINEAGE_PROTOCOL,
            }
            and (request.agent.role_family or "").casefold()
            == "evidence_retriever"
            else None
        )
        semantic_reasoner_question = (
            (
                hotpotqa_question_scope(request.problem)
                if semantic_reasoner_protocol
                in _HOTPOTQA_STRUCTURED_REASONER_PROTOCOLS
                else qa_question_scope(request.problem)
            )
            if semantic_reasoner_protocol is not None
            else None
        )
        semantic_evidence_retriever_question = (
            (
                hotpotqa_question_scope(request.problem)
                if semantic_evidence_retriever_protocol
                in _HOTPOTQA_STRUCTURED_REASONER_PROTOCOLS
                else qa_question_scope(request.problem)
            )
            if semantic_evidence_retriever_protocol is not None
            else None
        )
        semantic_upstream_tool_receipts = (
            self._direct_upstream_tool_receipts(request)
            if semantic_reasoner_protocol is not None
            else ()
        )
        retrieval_token = self._retrieval_completion_required.set(requires_retrieval)
        semantic_token = self._semantic_reasoner_question.set(
            semantic_reasoner_question
        )
        semantic_protocol_token = self._semantic_reasoner_protocol.set(
            semantic_reasoner_protocol
        )
        evidence_retriever_question_token = (
            self._semantic_evidence_retriever_question.set(
                semantic_evidence_retriever_question
            )
        )
        evidence_retriever_protocol_token = (
            self._semantic_evidence_retriever_protocol.set(
                semantic_evidence_retriever_protocol
            )
        )
        upstream_receipts_token = self._semantic_upstream_tool_receipts.set(
            semantic_upstream_tool_receipts
        )
        try:
            try:
                return await super().execute(request)
            except ReactExecutionError as exc:
                public_observations = self._continuation_observations(
                    exc.react_trace
                )
                state = self._required_evidence_state(
                    request,
                    public_observations,
                )
                if self._hotpot_tool_plan_exhausted(request, state):
                    remaining_tool_calls = max(
                        0,
                        self._max_tool_calls - state.dispatched_tool_calls,
                    )
                    trace = [dict(item) for item in exc.react_trace]
                    terminal_diagnosis = {
                        "observation_status": "budget_exhausted",
                        "public_error_code": "qa_retrieval_tool_plan_exhausted",
                        "tool_plan_exhausted": True,
                        "remaining_tool_calls": remaining_tool_calls,
                        "successful_tool_receipt_count": (
                            state.dispatched_tool_calls
                        ),
                        "successful_evidence_read_count": (
                            state.successful_read_count
                        ),
                    }
                    if trace:
                        trace[-1]["terminal_failure_diagnosis"] = (
                            terminal_diagnosis
                        )
                    else:  # pragma: no cover - bounded execution samples a turn
                        trace.append(terminal_diagnosis)
                    raise ReactExecutionError(
                        str(exc),
                        react_trace=tuple(trace),
                        tool_receipts=exc.tool_receipts,
                        model_calls=exc.model_calls,
                        tool_plan_exhausted=True,
                    ) from exc
                if not self._unified_factual_protocol(request):
                    raise
                exhausted = (
                    state.search_attempt_count
                    >= len(_FACTUAL_QA_RETRIEVAL_STRATEGIES)
                    or state.dispatched_tool_calls >= self._max_tool_calls
                )
                evidence_unresolved = (
                    state.successful_read_count == 0
                    or state.semantic_repair_kind in {"evidence", "coverage"}
                )
                if not exhausted or not evidence_unresolved:
                    if state.semantic_repair_kind is None:
                        raise
                    trace = [dict(item) for item in exc.react_trace]
                    if trace:
                        trace[-1]["repair_instruction"] = (
                            self._semantic_repair_instruction(
                                state.semantic_repair_kind
                            )
                        )
                    raise ReactExecutionError(
                        str(exc),
                        react_trace=tuple(trace),
                        tool_receipts=exc.tool_receipts,
                        model_calls=exc.model_calls,
                        tool_plan_exhausted=exc.tool_plan_exhausted,
                    ) from exc
                trace = [dict(item) for item in exc.react_trace]
                terminal_diagnosis = {
                    "observation_status": "budget_exhausted",
                    "public_error_code": _KNOWLEDGE_BASE_COVERAGE_FAILURE,
                    "tool_plan_exhausted": True,
                    "retrieval_attempt_count": state.search_attempt_count,
                    "retrieval_strategies_attempted": list(
                        _FACTUAL_QA_RETRIEVAL_STRATEGIES[
                            : state.search_attempt_count
                        ]
                    ),
                    "search_queries": list(state.search_queries),
                    "search_top_ks": list(state.search_top_ks),
                }
                if trace:
                    trace[-1]["repair_instruction"] = (
                        self._semantic_repair_instruction("coverage")
                    )
                    trace[-1]["terminal_failure_diagnosis"] = terminal_diagnosis
                else:  # pragma: no cover - bounded execution always samples a turn
                    trace.append(terminal_diagnosis)
                raise ReactExecutionError(
                    str(exc),
                    react_trace=tuple(trace),
                    tool_receipts=exc.tool_receipts,
                    model_calls=exc.model_calls,
                    tool_plan_exhausted=True,
                ) from exc
        finally:
            self._semantic_evidence_retriever_protocol.reset(
                evidence_retriever_protocol_token
            )
            self._semantic_evidence_retriever_question.reset(
                evidence_retriever_question_token
            )
            self._semantic_upstream_tool_receipts.reset(
                upstream_receipts_token
            )
            self._semantic_reasoner_protocol.reset(semantic_protocol_token)
            self._semantic_reasoner_question.reset(semantic_token)
            self._retrieval_completion_required.reset(retrieval_token)

    def _contract(
        self,
        request: AgentRequest,
        observations: list[Mapping[str, object]],
    ) -> str:
        contract = super()._contract(request, observations)
        evidence_state = self._required_evidence_state(request, observations)
        semantic_role = (request.agent.role_family or "").casefold()
        admitted_actions, completion_admitted = self._state_conditioned_action_domain(
            request,
            observations,
        )
        searched_passage_ids = evidence_state.latest_unread_passage_ids
        if self._task_type == "multi_hop_qa":
            guidance = SKILLFLOW_MULTI_HOP_QA_GUIDANCE
            if (
                request.semantic_protocol in _HOTPOTQA_STRUCTURED_REASONER_PROTOCOLS
                and semantic_role == "reasoner"
            ):
                guidance += " " + HOTPOTQA_VERIFIED_ANSWER_SLOT_GUIDANCE
            elif request.semantic_protocol == QA_VERIFIED_ANSWER_LINEAGE_PROTOCOL:
                guidance += " " + QA_VERIFIED_ANSWER_LINEAGE_GUIDANCE
        elif self._task_type == "factual_qa":
            guidance = SKILLFLOW_FACTUAL_QA_GUIDANCE
            if request.semantic_protocol == QA_VERIFIED_ANSWER_LINEAGE_PROTOCOL:
                guidance += " " + QA_VERIFIED_ANSWER_LINEAGE_GUIDANCE
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
        candidate_seen, routed_candidate = (
            self._role_conditional_upstream_candidate(request)
        )
        if candidate_seen:
            if completion_admitted and routed_candidate is not None:
                terminal_wire += (
                    " A routed semantic candidate has already been determined. "
                    "Set arguments.value to exactly "
                    f"<answer>{routed_candidate}</answer>; copy the candidate "
                    "character-for-character and do not reselect, canonicalize, "
                    "or rewrite it."
                )
            else:
                terminal_wire += (
                    " Routed semantic candidates are malformed or disagree, so do "
                    "not choose among them or emit a replacement semantic answer; "
                    "preserve the conflict for AgentGraph repair."
                )
        elif request.semantic_protocol in {
            *_HOTPOTQA_STRUCTURED_REASONER_PROTOCOLS,
            QA_VERIFIED_ANSWER_LINEAGE_PROTOCOL,
        } and semantic_role == "reasoner":
            if completion_admitted:
                terminal_wire += (
                    " As the Reasoner, arguments.value must contain exactly the six "
                    "structured fields question_scope, answer_slot, "
                    "evidence_propositions, multi_hop_chain, candidate_answer, and "
                    "evidence. Copy question_scope exactly from the original question. "
                    "Use answer_slot.proposition_index and answer_field to bind the "
                    "candidate to one explicit evidence proposition; candidate_answer "
                    "must equal that proposition's selected field. Set answer_type and "
                    "answer_cardinality from the original question and preserve its "
                    "qualifiers. For single-answer questions, return one minimal but "
                    "complete evidence-aligned referential surface rather than an alias "
                    "list, appositive gloss, or the question's answer-type head noun. In "
                    "a who-question with a possessive construction, exclude the "
                    "possessive marker and possessed attribute but retain the complete "
                    "possessor entity mention before it, including any title, honorific, "
                    "or name suffix present in the evidence."
                )
            else:
                terminal_wire += (
                    " The Reasoner remains responsible for semantic alignment after "
                    "evidence is read, but no semantic-answer field belongs in the "
                    "current Tool arguments."
                )
        elif request.semantic_protocol in {
            *_HOTPOTQA_STRUCTURED_REASONER_PROTOCOLS,
            QA_VERIFIED_ANSWER_LINEAGE_PROTOCOL,
        } and semantic_role == "evidence_retriever":
            if completion_admitted:
                if (
                    request.semantic_protocol
                    == HOTPOTQA_ROLE_CONDITIONAL_PROTOCOL
                ):
                    terminal_wire += (
                        " As the Evidence Retriever, arguments.value must be a "
                        "non-empty array of receipt-grounded citations. Each item "
                        "contains exactly passage_id and evidence_span copied from "
                        "one successful qa-retrieval read receipt. Do not add "
                        "entity_identity, target_relation, answer_type_constraint, "
                        "evidence_proposition, answer_slot, candidate_answer, or "
                        "final_answer; a downstream semantic producer owns "
                        "semantic alignment."
                    )
                else:
                    terminal_wire += (
                        " As the Evidence Retriever, arguments.value must contain "
                        "exactly question_scope, entity_identity, target_relation, "
                        "answer_type_constraint, evidence_proposition, evidence_span, "
                        "and passage_id. entity_identity contains exactly "
                        "question_surface and evidence_surface. question_surface is "
                        "the question-side entity/event anchor copied from the original "
                        "question, never a wh-word/wh-phrase, the whole question, or an "
                        "answer. evidence_surface is the coreferential surface of that "
                        "same anchor copied from the receipt-grounded exact evidence_span. "
                        "Cite one successful qa-retrieval read receipt and copy one exact "
                        "evidence span containing the anchor surface, exact predicate, and "
                        "proposition surfaces. A different question/evidence surface may "
                        "use the same read receipt's passage title for explicit alias "
                        "binding, but both surfaces must be supported by that receipt's "
                        "title/evidence_span."
                    )
                if (
                    request.semantic_protocol
                    != HOTPOTQA_ROLE_CONDITIONAL_PROTOCOL
                ):
                    if (
                        request.semantic_protocol
                        == QA_VERIFIED_ANSWER_LINEAGE_PROTOCOL
                    ):
                        terminal_wire += (
                            " For factual QA, when question_surface and evidence_surface "
                            "differ and question_surface, verbatim or after removing one "
                            "leading honorific, exactly equals that same read receipt's "
                            "passage title, evidence_surface must be a coreferential "
                            "identity surface present in both that title and the exact "
                            "evidence_span. The entity/event anchor may occupy the subject, "
                            "the object, or neither binary proposition argument. The "
                            "Reasoner alone owns relation binding, answer-slot binding, and "
                            "semantic answer selection."
                        )
                    else:
                        terminal_wire += (
                            " The anchor need not occupy a binary proposition argument; "
                            "the Reasoner owns answer-slot binding. If the anchor occupies "
                            "exactly one argument, the other argument must match the "
                            "question-only answer_type_constraint; it must not occupy both "
                            "arguments."
                        )
                    terminal_wire += (
                        " evidence_proposition records only receipt-grounded subject, "
                        "predicate, and object_or_attribute_value surfaces. Do not select "
                        "or emit candidate_answer, answer_slot, or final_answer."
                    )
            else:
                terminal_wire += (
                    " The Evidence Retriever owns only receipt-grounded evidence "
                    "provenance; no semantic or answer field belongs in the current "
                    "Tool arguments."
                    if request.semantic_protocol
                    == HOTPOTQA_ROLE_CONDITIONAL_PROTOCOL
                    else
                    " The Evidence Retriever owns only receipt-grounded entity, "
                    "relation, and evidence provenance; no answer field belongs "
                    "in the current Tool arguments."
                )
        elif request.semantic_protocol in {
            *_HOTPOTQA_STRUCTURED_REASONER_PROTOCOLS,
            QA_VERIFIED_ANSWER_LINEAGE_PROTOCOL,
        } and semantic_role == "verifier":
            terminal_wire += (
                " As the Verifier, check explicit retrieved evidence, entity-to-"
                "attribute binding, multi-hop completeness, and unchanged question "
                "scope; also check answer type/cardinality, a minimal but complete "
                "evidence-aligned referential surface, and alias binding. For a who-"
                "question with a possessive construction, reject a candidate that "
                "drops part of the possessor entity mention before the possessive "
                "marker, including a source title, honorific, or name suffix. Do not "
                + (
                    "replace the routed semantic producer's candidate. Return "
                    if request.semantic_protocol
                    == HOTPOTQA_ROLE_CONDITIONAL_PROTOCOL
                    else "replace the Reasoner's candidate. Return "
                )
                + "Candidate answer, the seven explicit boolean check fields, and "
                "Verification status. Every check field must be the literal boolean "
                "true or false, never the candidate text or an explanation. Set "
                "supported only when all checks pass; "
                "otherwise request repair without a substitute candidate."
            )
        elif request.is_format_predecessor:
            terminal_wire += (
                " As the direct predecessor of the Format Agent, put the brief "
                "answer span after `Candidate answer:` and the supporting "
                "retrieved passage span after `Evidence:` in arguments.value."
            )
        # DIRECT_REUSE: SkillFlow's bounded Agent resumes from public
        # Action--Observation history rather than discarding completed Tool
        # transitions after a rejected completion.  The HotpotQA wording below
        # narrows only the semantic repair obligation; the retained search/read
        # receipts and the next state-conditioned action remain that upstream
        # continuation boundary.
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
            if completion_admitted:
                if (
                    request.semantic_protocol
                    == HOTPOTQA_ROLE_CONDITIONAL_PROTOCOL
                    and semantic_role == "evidence_retriever"
                ):
                    evidence_continuation += (
                        "The required successful non-empty qa-retrieval reads are "
                        "present, so the next action must complete with the "
                        "receipt-grounded passage_id/evidence_span array. This turn "
                        "use kind=complete, name=complete, resource_id=null, and "
                        "skill_id=null; arguments must contain exactly one key, "
                        "value."
                    )
                else:
                    evidence_continuation += (
                        "The required successful non-empty qa-retrieval reads are present, "
                        "so the next action must complete after aligning every required "
                        "hop to the original answer slot. This turn use kind=complete, "
                        "name=complete, resource_id=null, and skill_id=null; arguments "
                        "must contain exactly one key, value, whose value is the full "
                        "structured semantic artifact. Do not use kind=completion, "
                        "name=answer, or resource_id=qa-retrieval."
                    )
            elif searched_passage_ids and admitted_actions == frozenset(
                {(QA_RETRIEVAL_TOOL_ID, "read")}
            ):
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
                if self._unified_factual_protocol(request):
                    strategy = self._factual_retrieval_strategy(
                        evidence_state.search_attempt_count
                    )
                    expected_top_k = self._factual_search_limit(
                        evidence_state.search_attempt_count
                    )
                    evidence_continuation += (
                        "The next action must be qa-retrieval search using the "
                        f"bounded retrieval strategy `{strategy}` at attempt "
                        f"{evidence_state.search_attempt_count + 1}, with limit "
                        f"exactly {expected_top_k}. Preserve the target relation "
                        "and answer slot while changing the entity query according "
                        "to this strategy. For spelling_normalization, normalize "
                        "only the entity spelling; for alias_expansion, search an "
                        "evidence-supported alias; for entity_disambiguation, add "
                        "a relation or identifying qualifier; for query_rewriting, "
                        "rewrite the entity-and-relation query without narrowing "
                        "the question scope. The normalized query must differ from "
                        "all prior search queries: "
                        + json.dumps(
                            list(evidence_state.normalized_search_queries),
                            ensure_ascii=False,
                        )
                        + ". Then read one exact returned passage_id. Search "
                        "arguments contain exactly query and limit; do not put the "
                        "eventual semantic artifact in Tool arguments. If all "
                        "bounded strategies fail to bind entity identity and target "
                        "relation to a successful Tool receipt, report "
                        "knowledge_base_coverage_failure and never guess or fabricate "
                        "evidence."
                    )
                else:
                    evidence_continuation += (
                        "The next action must be qa-retrieval search with a concise "
                        "entity-and-relation query for a missing hop. Then read an exact returned "
                        "passage_id. Search arguments contain exactly query and limit; "
                        "never copy JSON-Schema properties/additionalProperties or the "
                        "eventual semantic artifact into those arguments. Set query to "
                        "one non-empty focused entity-and-relation string selected from "
                        "the original question or newest read evidence. For HotpotQA, "
                        "set limit to exactly 10 so the bounded continuation can inspect "
                        "two distinct ranked passages from the same search result. Keep "
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

    @staticmethod
    def _role_conditional_evidence_completion_issue(
        *,
        artifact: str,
        tool_receipts: Sequence[Mapping[str, object]],
    ) -> str | None:
        """Validate receipt-grounded citations without semantic interpretation."""

        from .agent_workflow_env import (
            AgentWorkflowEnv,
            _evidence_span_matches_read,
        )

        try:
            citations = json.loads(artifact)
        except (TypeError, ValueError):
            return "Evidence Retriever citations must be one JSON array"
        if not isinstance(citations, list) or not citations:
            return "Evidence Retriever citations must be a non-empty JSON array"
        for index, citation in enumerate(citations):
            if not isinstance(citation, Mapping) or set(citation) != {
                "passage_id",
                "evidence_span",
            }:
                return (
                    "Evidence Retriever citation must contain exactly "
                    f"passage_id and evidence_span; citation_index={index}"
                )
            passage_id = citation.get("passage_id")
            evidence_span = citation.get("evidence_span")
            if any(
                not isinstance(value, str)
                or not value.strip()
                or value != value.strip()
                for value in (passage_id, evidence_span)
            ):
                return (
                    "Evidence Retriever citation passage_id and evidence_span "
                    f"must be non-empty trimmed text; citation_index={index}"
                )
            assert isinstance(passage_id, str)
            assert isinstance(evidence_span, str)
            matched_read_text: str | None = None
            for receipt in tool_receipts:
                if not AgentWorkflowEnv._successful_read_receipt(
                    receipt,
                    QA_RETRIEVAL_TOOL_ID,
                ):
                    continue
                receipt_request = receipt.get("request")
                if not isinstance(receipt_request, Mapping):
                    continue
                receipt_arguments = receipt_request.get("arguments")
                if (
                    not isinstance(receipt_arguments, Mapping)
                    or receipt_arguments.get("passage_id") != passage_id
                ):
                    continue
                matched_read_text = AgentWorkflowEnv._successful_read_text(
                    receipt,
                    QA_RETRIEVAL_TOOL_ID,
                )
                if matched_read_text is not None:
                    break
            if matched_read_text is None:
                return (
                    "Evidence Retriever citation passage_id has no matching "
                    "successful qa-retrieval read receipt; "
                    f"citation_index={index}"
                )
            if not _evidence_span_matches_read(
                evidence_span,
                matched_read_text,
            ):
                return (
                    "Evidence Retriever citation evidence_span has no "
                    "typography-canonical lexical match in its qa-retrieval "
                    f"read receipt; citation_index={index}"
                )
        return None

    @staticmethod
    def _evidence_retriever_completion_issue(
        *,
        original_question: str,
        artifact: str,
        tool_receipts: Sequence[Mapping[str, object]],
    ) -> str | None:
        """Validate one answer-free Retriever artifact against one read receipt.

        An exact same-receipt title binding can ground a distinct question and
        evidence entity surface without weakening the contextual-anchor path.
        """

        from .agent_workflow_env import (
            AgentWorkflowEnv,
            _PERSON_TITLE_PATTERN,
            _canonical_evidence_text,
            _evidence_span_matches_read,
        )

        required_fields = (
            "question_scope",
            "entity_identity",
            "target_relation",
            "answer_type_constraint",
            "evidence_proposition",
            "evidence_span",
            "passage_id",
        )
        fields, issue = AgentWorkflowEnv._structured_semantic_fields(
            artifact,
            required_fields,
        )
        if issue is not None or fields is None:
            return issue or "Evidence Retriever artifact is missing"
        if fields.get("question_scope") != original_question:
            return "Evidence Retriever question_scope must equal the original question"

        passage_id = fields.get("passage_id")
        evidence_span = fields.get("evidence_span")
        target_relation = fields.get("target_relation")
        answer_type_constraint = fields.get("answer_type_constraint")
        evidence_proposition = fields.get("evidence_proposition")
        identity = fields.get("entity_identity")
        if (
            not isinstance(passage_id, str)
            or not passage_id.strip()
            or passage_id != passage_id.strip()
            or not isinstance(evidence_span, str)
            or not evidence_span.strip()
            or evidence_span != evidence_span.strip()
            or not isinstance(target_relation, str)
            or not target_relation.strip()
            or target_relation != target_relation.strip()
        ):
            return (
                "Evidence Retriever passage_id, target_relation, and evidence_span "
                "must be non-empty trimmed text"
            )
        if not isinstance(identity, Mapping) or set(identity) != {
            "question_surface",
            "evidence_surface",
        }:
            return (
                "Evidence Retriever entity_identity must contain exactly "
                "question_surface and evidence_surface"
            )
        question_surface = identity.get("question_surface")
        evidence_surface = identity.get("evidence_surface")
        if any(
            not isinstance(value, str)
            or not value.strip()
            or value != value.strip()
            for value in (question_surface, evidence_surface)
        ):
            return "Evidence Retriever entity surfaces must be non-empty trimmed text"
        assert isinstance(question_surface, str)
        assert isinstance(evidence_surface, str)

        expected_answer_type = qa_answer_type_constraint(original_question)
        if answer_type_constraint != expected_answer_type:
            return (
                "Evidence Retriever answer_type_constraint must equal the "
                "question-only qa_answer_type_constraint"
            )
        if not isinstance(evidence_proposition, Mapping) or set(
            evidence_proposition
        ) != {
            "subject",
            "predicate",
            "object_or_attribute_value",
        }:
            return (
                "Evidence Retriever evidence_proposition must contain exactly "
                "subject, predicate, and object_or_attribute_value"
            )
        proposition_values = tuple(
            evidence_proposition.get(field)
            for field in ("subject", "predicate", "object_or_attribute_value")
        )
        if any(
            not isinstance(value, str)
            or not value.strip()
            or value != value.strip()
            for value in proposition_values
        ):
            return (
                "Evidence Retriever evidence proposition surfaces must be "
                "non-empty trimmed text"
            )
        proposition_subject, proposition_predicate, proposition_object = (
            proposition_values
        )
        assert isinstance(proposition_subject, str)
        assert isinstance(proposition_predicate, str)
        assert isinstance(proposition_object, str)

        cited_read_text: str | None = None
        cited_read_title: str | None = None
        for receipt in tool_receipts:
            if not AgentWorkflowEnv._successful_read_receipt(
                receipt,
                QA_RETRIEVAL_TOOL_ID,
            ):
                continue
            receipt_request = receipt.get("request")
            assert isinstance(receipt_request, Mapping)
            receipt_arguments = receipt_request.get("arguments")
            if not isinstance(receipt_arguments, Mapping):
                continue
            request_passage_id = receipt_arguments.get("passage_id")
            if request_passage_id != passage_id:
                continue
            result = receipt.get("result")
            assert isinstance(result, Mapping)
            value = result.get("value", result)
            assert isinstance(value, Mapping)
            passage = value.get("passage")
            assert isinstance(passage, Mapping)
            result_ids = (
                value.get("passage_id"),
                passage.get("passage_id"),
            )
            if any(
                result_id is not None and result_id != passage_id
                for result_id in result_ids
            ):
                continue
            cited_read_text = AgentWorkflowEnv._successful_read_text(
                receipt,
                QA_RETRIEVAL_TOOL_ID,
            )
            if cited_read_text is not None:
                raw_title = passage.get("title")
                if isinstance(raw_title, str) and raw_title.strip():
                    cited_read_title = raw_title.strip()
                break
        if cited_read_text is None:
            return (
                "Evidence Retriever passage_id has no matching successful "
                "qa-retrieval read receipt"
            )
        if not _evidence_span_matches_read(evidence_span, cited_read_text):
            return (
                "Evidence Retriever evidence_span has no typography-canonical "
                "lexical match in the cited qa-retrieval read receipt"
            )

        canonical_question = _canonical_evidence_text(original_question)
        canonical_span = _canonical_evidence_text(evidence_span)
        canonical_title = (
            _canonical_evidence_text(cited_read_title)
            if cited_read_title is not None
            else ""
        )
        canonical_question_surface = _canonical_evidence_text(question_surface)
        canonical_evidence_surface = _canonical_evidence_text(evidence_surface)
        canonical_relation = _canonical_evidence_text(target_relation)
        canonical_subject = _canonical_evidence_text(proposition_subject)
        canonical_predicate = _canonical_evidence_text(proposition_predicate)
        canonical_object = _canonical_evidence_text(proposition_object)
        honorific_normalized_question = re.sub(
            rf"^(?:{_PERSON_TITLE_PATTERN})\s+",
            "",
            canonical_question_surface,
            count=1,
            flags=re.IGNORECASE,
        ).strip()
        question_title_binding = bool(
            canonical_title
            and (
                canonical_question_surface == canonical_title
                or (
                    honorific_normalized_question != canonical_question_surface
                    and honorific_normalized_question == canonical_title
                )
            )
        )
        if canonical_question_surface not in canonical_question:
            return (
                "Evidence Retriever entity_identity.question_surface does not "
                "occur in the original question"
            )
        if canonical_evidence_surface not in canonical_span:
            return (
                "Evidence Retriever entity_identity.evidence_surface does not "
                "occur in evidence_span"
            )
        if canonical_question_surface == canonical_question:
            return (
                "Evidence Retriever entity_identity.question_surface must be a "
                "concise entity mention, not the whole original question"
            )
        question_anchor_tokens = re.findall(
            r"\w+",
            question_surface.casefold(),
            flags=re.UNICODE,
        )
        if (
            question_anchor_tokens
            and question_anchor_tokens[0] in _QUESTION_ANCHOR_WH_WORDS
        ):
            return (
                "Evidence Retriever entity_identity.question_surface must be a "
                "question-side entity/event anchor, not a wh-word or wh-phrase"
            )
        if canonical_evidence_surface == canonical_span:
            return (
                "Evidence Retriever entity_identity.evidence_surface must be a "
                "concise entity mention, not the whole evidence_span"
            )
        if canonical_relation not in canonical_span:
            return "Evidence Retriever target_relation does not occur in evidence_span"
        if canonical_question_surface == canonical_relation or (
            canonical_evidence_surface == canonical_relation
        ):
            return (
                "Evidence Retriever entity surface must not be the relation "
                "predicate surface"
            )
        for field_name, proposition_surface in (
            ("subject", canonical_subject),
            ("predicate", canonical_predicate),
            ("object_or_attribute_value", canonical_object),
        ):
            if proposition_surface not in canonical_span:
                return (
                    "Evidence Retriever evidence_proposition."
                    f"{field_name} does not occur in evidence_span"
                )
        if canonical_predicate != canonical_relation:
            return (
                "Evidence Retriever evidence_proposition.predicate must equal "
                "target_relation"
            )

        if (
            question_title_binding
            and canonical_question_surface != canonical_evidence_surface
            and canonical_evidence_surface not in canonical_title
        ):
            return (
                "Evidence Retriever entity_identity.evidence_surface is not "
                "supported by the cited passage title identity chain"
            )

        entity_in_subject = canonical_evidence_surface in canonical_subject
        entity_in_object = canonical_evidence_surface in canonical_object
        if entity_in_subject and entity_in_object:
            return (
                "Evidence Retriever evidence proposition must not bind "
                "entity_identity.evidence_surface to both relation arguments"
            )
        elif entity_in_subject != entity_in_object:
            open_argument = (
                proposition_object if entity_in_subject else proposition_subject
            )
            answer_type_issue = _answer_surface_type_issue(
                expected_type=expected_answer_type,
                surface=open_argument,
            )
            if answer_type_issue is not None:
                return answer_type_issue

        if (
            canonical_question_surface != canonical_evidence_surface
            and canonical_question_surface not in canonical_span
        ):
            canonical_receipt_identity_surface = " ".join(
                part for part in (canonical_title, canonical_span) if part
            )
            explicit_receipt_binding = not (
                canonical_question_surface
                not in canonical_receipt_identity_surface
                or canonical_evidence_surface not in canonical_receipt_identity_surface
            )

            # PROJECT_NECESSARY_ADAPTATION: preserve SkillFlow's exact read-
            # receipt boundary while normalizing only one leading linguistic
            # honorific already enumerated by the shared semantic validator.
            title_identity_binding = bool(
                question_title_binding
                and canonical_evidence_surface in canonical_title
            )
            if not explicit_receipt_binding and not title_identity_binding:
                return (
                    "Evidence Retriever alias identity lacks an explicit binding "
                    "between question_surface and evidence_surface in the cited "
                    "read receipt title/evidence_span"
                )
        return None

    def _completion_error(
        self,
        *,
        action: StructuredAction,
        artifact: str,
        tool_receipts: list[dict[str, object]],
    ) -> str | None:
        del action

        semantic_protocol = self._semantic_reasoner_protocol.get()
        unified_factual = (
            semantic_protocol == QA_VERIFIED_ANSWER_LINEAGE_PROTOCOL
            and self._task_type == "factual_qa"
        )
        search_attempt_count = sum(
            1
            for receipt in tool_receipts
            if receipt.get("tool_id") == QA_RETRIEVAL_TOOL_ID
            and isinstance(receipt.get("request"), Mapping)
            and receipt["request"].get("action") == "search"
        )
        retrieval_budget_exhausted = (
            len(tool_receipts) >= self._max_tool_calls
            or search_attempt_count >= len(_FACTUAL_QA_RETRIEVAL_STRATEGIES)
        )
        upstream_tool_receipts = list(
            self._semantic_upstream_tool_receipts.get()
        )
        evidence_tool_receipts = [
            *tool_receipts,
            *upstream_tool_receipts,
        ]

        # DIRECT_REUSE: SkillFlow training/environment.py::step admits the
        # terminal answer after at least one non-answer Tool turn.  Preserve
        # that historical ``required_tool_call`` boundary, including a failed
        # receipt, so a Tool outage cannot discard a usable upstream answer.
        # ``required_evidence`` is an explicit stricter experiment condition:
        # only a successful read receipt carrying non-empty public text counts.
        if self._retrieval_completion_required.get():
            retrieval_admitted = False
            for receipt in evidence_tool_receipts:
                if receipt.get("tool_id") != QA_RETRIEVAL_TOOL_ID:
                    continue
                receipt_request = receipt.get("request")
                if not isinstance(receipt_request, Mapping) or receipt_request.get(
                    "action"
                ) not in {"search", "read"}:
                    continue
                if self._completion_policy == "required_tool_call":
                    retrieval_admitted = True
                    break
                result = receipt.get("result")
                if (
                    not isinstance(result, Mapping)
                    or receipt.get("error_type") is not None
                ):
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
                    retrieval_admitted = True
                    break
            if not retrieval_admitted:
                if self._completion_policy == "required_tool_call":
                    return "qa_completion_requires_retrieval_dispatch"
                if unified_factual and retrieval_budget_exhausted:
                    return _KNOWLEDGE_BASE_COVERAGE_FAILURE
                return "qa_completion_requires_successful_read_evidence"

        evidence_retriever_question = (
            self._semantic_evidence_retriever_question.get()
        )
        if evidence_retriever_question is not None:
            retriever_protocol = self._semantic_evidence_retriever_protocol.get()
            issue = (
                self._role_conditional_evidence_completion_issue(
                    artifact=artifact,
                    tool_receipts=tool_receipts,
                )
                if retriever_protocol == HOTPOTQA_ROLE_CONDITIONAL_PROTOCOL
                else self._evidence_retriever_completion_issue(
                    original_question=evidence_retriever_question,
                    artifact=artifact,
                    tool_receipts=tool_receipts,
                )
            )
            if issue is None:
                return None
            # A receipt/span/alias-lineage failure needs another search/read.
            # A malformed or lexically misaligned Retriever field can be
            # corrected against the already-successful read.  Route that case
            # through the existing structured-artifact repair branch so the
            # SkillFlow Action--Observation continuation does not mislabel a
            # usable passage as database coverage failure.
            structured_repair = (
                retriever_protocol == HOTPOTQA_ROLE_CONDITIONAL_PROTOCOL
                or issue.startswith(
                    (
                        "artifact is empty",
                        "labelled artifact",
                        "artifact must",
                        "field ",
                        "Evidence Retriever question_scope",
                        (
                            "Evidence Retriever passage_id, target_relation, and "
                            "evidence_span"
                        ),
                        "Evidence Retriever entity_identity must contain",
                        "Evidence Retriever entity surfaces",
                        (
                            "Evidence Retriever evidence_span has no "
                            "typography-canonical lexical match"
                        ),
                        "Evidence Retriever entity_identity.question_surface",
                        "Evidence Retriever entity_identity.evidence_surface",
                        "Evidence Retriever target_relation",
                        "Evidence Retriever answer_type_constraint",
                        "Evidence Retriever evidence_proposition must contain",
                        "Evidence Retriever evidence proposition surfaces",
                        "Evidence Retriever entity surface must not be",
                        "Evidence Retriever evidence_proposition.",
                        "Evidence Retriever evidence_proposition.predicate",
                        "Evidence Retriever evidence proposition must not bind",
                        (
                            "Evidence Retriever entity_identity.evidence_surface is not "
                            "supported by the cited passage title identity chain"
                        ),
                    )
                )
            )
            if retriever_protocol in _HOTPOTQA_STRUCTURED_REASONER_PROTOCOLS:
                prefix = (
                    _HOTPOTQA_SEMANTIC_STRUCTURE_ERROR_PREFIX
                    if structured_repair
                    else _HOTPOTQA_SEMANTIC_EVIDENCE_ERROR_PREFIX
                )
            else:
                prefix = (
                    _QA_SEMANTIC_STRUCTURE_ERROR_PREFIX
                    if structured_repair
                    else _QA_SEMANTIC_EVIDENCE_ERROR_PREFIX
                )
            return prefix + " " + issue

        original_question = self._semantic_reasoner_question.get()
        if original_question is None:
            return None
        # NECESSARY_ADAPTATION: completion admission in the generic bounded
        # executor has no AgentRequest parameter.  Reuse AgentWorkflowEnv's
        # semantic artifact parser/validator under a request-scoped ContextVar
        # so an invalid Reasoner completion becomes public repair feedback in
        # the same ReAct loop instead of breaking the outer semantic lineage.
        from .agent_workflow_env import AgentWorkflowEnv

        reasoner_kwargs: dict[str, object] = {
            "original_question": original_question,
        }
        if unified_factual:
            reasoner_kwargs.update(
                minimum_evidence_propositions=1,
                minimum_reasoning_steps=1,
            )
        candidate, issue = AgentWorkflowEnv._reasoner_candidate(
            artifact,
            **reasoner_kwargs,
        )
        if issue is not None or candidate is None:
            detail = issue or "Reasoner candidate_answer is missing"
            evidence_binding_issue = (
                "must occur verbatim in the selected evidence_span" in detail
            )
            if semantic_protocol == QA_VERIFIED_ANSWER_LINEAGE_PROTOCOL:
                prefix = (
                    _QA_SEMANTIC_EVIDENCE_ERROR_PREFIX
                    if evidence_binding_issue
                    else _QA_SEMANTIC_STRUCTURE_ERROR_PREFIX
                )
            else:
                prefix = (
                    _HOTPOTQA_SEMANTIC_EVIDENCE_ERROR_PREFIX
                    if evidence_binding_issue
                    else _HOTPOTQA_SEMANTIC_STRUCTURE_ERROR_PREFIX
                )
            return prefix + " " + detail
        read_evidence_texts = tuple(
            text
            for receipt in evidence_tool_receipts
            if isinstance(receipt, Mapping)
            for text in (
                AgentWorkflowEnv._successful_read_text(
                    receipt,
                    QA_RETRIEVAL_TOOL_ID,
                ),
            )
            if text is not None
        )
        provenance_issue = AgentWorkflowEnv._reasoner_evidence_provenance_issue(
            artifact,
            read_evidence_texts,
            require_answer_binding=(
                semantic_protocol == QA_VERIFIED_ANSWER_LINEAGE_PROTOCOL
            ),
        )
        if provenance_issue is not None:
            entity_binding_repair = (
                "no deterministic entity binding" in provenance_issue
            )
            if entity_binding_repair:
                prefix = (
                    _QA_SEMANTIC_STRUCTURE_ERROR_PREFIX
                    if semantic_protocol == QA_VERIFIED_ANSWER_LINEAGE_PROTOCOL
                    else _HOTPOTQA_SEMANTIC_STRUCTURE_ERROR_PREFIX
                )
                return prefix + " " + provenance_issue
            if unified_factual and retrieval_budget_exhausted:
                return _KNOWLEDGE_BASE_COVERAGE_FAILURE
            prefix = (
                _QA_SEMANTIC_EVIDENCE_ERROR_PREFIX
                if semantic_protocol == QA_VERIFIED_ANSWER_LINEAGE_PROTOCOL
                else _HOTPOTQA_SEMANTIC_EVIDENCE_ERROR_PREFIX
            )
            return prefix + " " + provenance_issue
        return None

    def _tool_action_error(
        self,
        *,
        request: AgentRequest,
        action: StructuredAction,
        observations: list[Mapping[str, object]],
    ) -> str | None:
        state = self._required_evidence_state(request, observations)
        if (
            self._unified_factual_protocol(request)
            and action.kind is ActionKind.TOOL
            and action.resource_id == QA_RETRIEVAL_TOOL_ID
            and action.name == "search"
        ):
            if (
                state.search_attempt_count
                >= len(_FACTUAL_QA_RETRIEVAL_STRATEGIES)
                or state.dispatched_tool_calls >= self._max_tool_calls
            ):
                return _KNOWLEDGE_BASE_COVERAGE_FAILURE
            arguments = action.arguments
            if isinstance(arguments, dict):
                query = arguments.get("query")
                limit = arguments.get("limit")
                if isinstance(query, str) and query.strip():
                    normalized_query = _normalized_retrieval_query(query)
                    if normalized_query in state.normalized_search_queries:
                        return "qa_retrieval_duplicate_normalized_query"
                expected_limit = self._factual_search_limit(
                    state.search_attempt_count
                )
                if type(limit) is int and limit != expected_limit:
                    return (
                        "qa_retrieval_top_k_mismatch: expected "
                        f"{expected_limit} for attempt "
                        f"{state.search_attempt_count + 1}"
                    )
        admitted_actions, completion_admitted = self._state_conditioned_action_domain(
            request,
            observations,
        )
        if state.required and action.kind is ActionKind.TOOL:
            if admitted_actions is not None and (
                action.resource_id,
                action.name,
            ) not in admitted_actions:
                if completion_admitted:
                    return "qa_required_evidence_next_action_complete"
                if (
                    self._unified_factual_protocol(request)
                    and not admitted_actions
                ):
                    return _KNOWLEDGE_BASE_COVERAGE_FAILURE
                expected_action = (
                    next(iter(admitted_actions))[1] if admitted_actions else "complete"
                )
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

        admitted_passage_ids = set(state.latest_unread_passage_ids)
        if not admitted_passage_ids:
            return "qa_read_requires_successful_search"
        if passage_id not in admitted_passage_ids:
            return "qa_read_passage_id_not_from_search"
        return None


@dataclass(frozen=True)
class QAStructuredReasoningExecutionAdapter:
    """Apply SkillFlow's existing semantic completion schema to QA reasoning.

    A reasoning-mode Reasoner consumes routed Retriever evidence without
    executing another Tool loop. Its semantic artifact nevertheless has the
    same public contract as a ReAct Reasoner's completion. Reuse the exact
    request-scoped completion schema already owned by
    :class:`QARetrievalReactExecutionAdapter` and send it through SkillFlow's
    strict ``response_schema`` provider boundary. Other roles and protocols
    retain the unchanged reasoning gateway path.
    """

    gateway: AgentGateway
    schema_source: QARetrievalReactExecutionAdapter

    async def execute(self, request: AgentRequest) -> GatewayResponse:
        mode = getattr(
            request.agent.execution_mode,
            "value",
            request.agent.execution_mode,
        )
        if mode != "reasoning":
            raise ValueError(
                "structured QA reasoning adapter requires execution_mode reasoning"
            )
        semantic_role = (request.agent.role_family or "").casefold()
        if semantic_role != "reasoner" or request.semantic_protocol not in {
            *_HOTPOTQA_STRUCTURED_REASONER_PROTOCOLS,
            QA_VERIFIED_ANSWER_LINEAGE_PROTOCOL,
        }:
            return await self.gateway.generate(request)

        completion_arguments = self.schema_source._completion_arguments_schema(
            request
        )
        value_schema = completion_arguments.get("properties", {}).get("value")
        if not isinstance(value_schema, Mapping):
            raise ValueError(
                "QA semantic completion schema has no arguments.value object"
            )
        # DIRECT_REUSE: SkillFlow ``skillev/runtime/openai_provider.py`` sends
        # each request's response_schema through strict OpenAI JSON Schema.
        # The project Gateway already mirrors that boundary; only the existing
        # QA completion value schema is projected from ReAct to reasoning here.
        model_metadata = {
            **dict(request.model.metadata),
            "response_json_schema": json.dumps(
                value_schema,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ),
        }
        constrained_request = replace(
            request,
            model=replace(request.model, metadata=model_metadata),
        )
        return await self.gateway.generate(constrained_request)


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
    "QAStructuredReasoningExecutionAdapter",
    "QARetrievalToolBackend",
    "QAReadToolBackend",
    "QA_RETRIEVAL_TOOL_ID",
    "QA_VERIFIED_ANSWER_LINEAGE_PROTOCOL",
    "QASearchToolBackend",
    "build_qa_tool_registry",
    "open_qa_tool_registry",
    "open_provided_context_qa_tool_registry",
]
