from __future__ import annotations

from dataclasses import dataclass, replace
import json
from pathlib import Path
from threading import Event
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from src.interactive.agent_graph import AgentNode
from src.interactive.agent_runtime import (
    AgentRequest,
    AgentResponse,
    ExecutionPhase,
    UpstreamMessage,
)
from src.interactive.model_registry import ModelSpec, ProviderSpec
from src.interactive.react_execution import ReactExecutionError
from src.interactive.qa_tool_adapter import (
    QARetrievalReactExecutionAdapter,
    QA_RETRIEVAL_TOOL_ID,
    QA_VERIFIED_ANSWER_LINEAGE_PROTOCOL,
    _factual_strategy_semantics_verified,
    _location_containment_lineage_issue,
    _public_search_candidate_compatibility,
    _query_replaces_relation_surface,
    _question_entity_anchor_tokens,
    build_qa_tool_registry,
    open_qa_tool_registry,
    open_provided_context_qa_tool_registry,
)
from src.interactive.tool_runtime import ActionKind, StructuredAction, ToolRequest


@dataclass(frozen=True)
class FakeManifest:
    corpus_name: str = "public-wikipedia"
    corpus_version: str = "snapshot-2026-08"
    index_id: str = "frozen-index-v1"
    format: str = "skillev-public-retrieval-index@2"
    retrieval_backend: str = "sqlite-fts5-lexical"


@dataclass(frozen=True)
class FakeHit:
    passage_id: str
    document_id: str
    title: str
    snippet: str
    rank: int


@dataclass(frozen=True)
class FakePassage:
    passage_id: str
    document_id: str
    title: str
    text: str


class FakeReadReceiptText(str):
    passage_title: str

    def __new__(cls, text: str, *, passage_title: str) -> "FakeReadReceiptText":
        value = super().__new__(cls, text)
        value.passage_title = passage_title
        return value


class FakeIndex:
    manifest = FakeManifest()

    def __init__(self) -> None:
        self.search_calls: list[tuple[str, int]] = []
        self.read_calls: list[str] = []
        self.closed = False

    def search(self, query: str, *, limit: int) -> tuple[FakeHit, ...]:
        self.search_calls.append((query, limit))
        return (
            FakeHit("p1", "d1", "Ada Lovelace", "Public evidence.", 1),
        )

    def read(self, passage_id: str) -> FakePassage:
        self.read_calls.append(passage_id)
        return FakePassage(passage_id, "d1", "Ada Lovelace", "Full public text.")

    def close(self) -> None:
        self.closed = True


class QAToolAdapterTests(unittest.IsolatedAsyncioTestCase):
    async def test_model_visible_continuation_collapses_only_duplicate_errors(
        self,
    ) -> None:
        error = {
            "observation_status": "schema_invalid",
            "public_error_code": (
                "qa_semantic_artifact_invalid: repair answer binding"
            ),
        }
        success = {
            "observation_status": "success",
            "result": {"operation": "read", "passage_id": "p1"},
        }

        visible = QARetrievalReactExecutionAdapter._model_visible_observations(
            [error, dict(error), success, error, dict(error)]
        )

        self.assertEqual(3, len(visible))
        self.assertEqual("success", visible[1]["observation_status"])
        self.assertIn("repair_instruction", visible[0])
        self.assertIn("repair_instruction", visible[2])
        self.assertEqual(2, visible[0]["repeat_count"])
        self.assertEqual(2, visible[2]["repeat_count"])

    def test_answer_slot_binding_feedback_is_field_specific_and_completion_only(
        self,
    ) -> None:
        mismatch = (
            "qa_semantic_artifact_invalid: Reasoner "
            "answer_slot.answer_field selects 'subject', but candidate_answer "
            "exactly matches the selected proposition field "
            "'object_or_attribute_value'; set answer_field to the proposition "
            "field containing candidate_answer"
        )

        repair = (
            QARetrievalReactExecutionAdapter._public_semantic_repair_instruction(
                mismatch
            )
        )

        self.assertIsNotNone(repair)
        assert repair is not None
        self.assertIn("Preserve every successful qa-retrieval read receipt", repair)
        self.assertIn("set answer_slot.proposition_index", repair)
        self.assertIn(
            "answer_slot.answer_field to that same proposition field",
            repair,
        )
        self.assertIn("Do not search or read again", repair)
        self.assertNotIn("York", repair)

    def test_temporal_answer_slot_feedback_preserves_normalized_candidate(
        self,
    ) -> None:
        mismatch = (
            "qa_semantic_artifact_invalid: Reasoner "
            "answer_slot.answer_field selects 'subject', but candidate_answer "
            "is the verified year-to-decade normalization of the selected "
            "proposition field 'object_or_attribute_value'; set answer_field "
            "to that proposition field"
        )

        repair = (
            QARetrievalReactExecutionAdapter._public_semantic_repair_instruction(
                mismatch
            )
        )

        self.assertIsNotNone(repair)
        assert repair is not None
        self.assertIn("source year or date", repair)
        self.assertIn("Preserve the normalized candidate_answer", repair)
        self.assertIn("Do not search or read again", repair)
        self.assertNotIn("equals candidate_answer exactly", repair)

    def test_entity_identity_feedback_repairs_only_receipt_grounded_fields(
        self,
    ) -> None:
        adapter = QARetrievalReactExecutionAdapter
        evidence_surface = adapter._public_semantic_repair_instruction(
            "qa_semantic_artifact_invalid: Evidence Retriever "
            "entity_identity.evidence_surface does not occur in evidence_span"
        )
        question_surface = adapter._public_semantic_repair_instruction(
            "qa_semantic_artifact_invalid: Evidence Retriever "
            "entity_identity.question_surface does not occur in the original "
            "question"
        )
        short_answer_surface = adapter._public_semantic_repair_instruction(
            "qa_semantic_artifact_invalid: Evidence Retriever answer-bearing "
            "entity surface is a strict subset of the resolved passage-title "
            "identity"
        )
        expanded_identity = adapter._public_semantic_repair_instruction(
            "qa_semantic_artifact_invalid: Evidence Retriever "
            "entity_identity.evidence_surface is not supported by the cited "
            "passage title identity chain"
        )
        proposition_binding = adapter._public_semantic_repair_instruction(
            "qa_semantic_artifact_invalid: Reasoner requested-relation "
            "proposition has no deterministic entity binding"
        )

        self.assertIsNotNone(evidence_surface)
        self.assertIsNotNone(question_surface)
        self.assertIsNotNone(short_answer_surface)
        self.assertIsNotNone(expanded_identity)
        self.assertIsNotNone(proposition_binding)
        assert evidence_surface is not None
        assert question_surface is not None
        assert short_answer_surface is not None
        assert expanded_identity is not None
        assert proposition_binding is not None
        self.assertIn("expand evidence_span only as needed", evidence_surface)
        self.assertIn("same read receipt", evidence_surface)
        self.assertIn("span contiguous", evidence_surface)
        self.assertIn("Repair only question_surface", question_surface)
        self.assertIn("unchanged original question", question_surface)
        self.assertIn(
            "every successful qa-retrieval read receipt",
            short_answer_surface,
        )
        self.assertIn(
            "complete public passage-title identity",
            short_answer_surface,
        )
        self.assertIn(
            "expanded full-name surface",
            short_answer_surface,
        )
        self.assertIn("admitted bounded retrieval", short_answer_surface)
        self.assertIn(
            "set evidence_surface to the complete public passage-title identity",
            expanded_identity,
        )
        self.assertIn(
            "expanded full-name proposition argument",
            expanded_identity,
        )
        self.assertIn(
            "every successful qa-retrieval read receipt",
            proposition_binding,
        )
        self.assertIn("do not add a search or read", proposition_binding)
        for repair in (
            evidence_surface,
            question_surface,
            short_answer_surface,
            expanded_identity,
            proposition_binding,
        ):
            self.assertNotIn("Sinclair Lewis", repair)
            self.assertNotIn("Judi Dench", repair)
            self.assertNotIn("York", repair)

    def test_repeated_structural_repair_stays_on_preserved_evidence(self) -> None:
        adapter = QARetrievalReactExecutionAdapter(
            gateway=SimpleNamespace(generate=lambda request: None),
            tool_registry=build_qa_tool_registry(FakeIndex()),
            max_turns=8,
            max_tool_calls=12,
            task_type="factual_qa",
            completion_policy="required_evidence",
        )
        request = AgentRequest(
            request_id="trivia:bounded-structural-repair",
            run_id="trivia",
            graph_revision=1,
            problem="In which decade did Billboard first publish an American hit chart?",
            agent=AgentNode(
                "reasoner",
                "model",
                "retrieve evidence and bind the requested relation",
                role_family="reasoner",
                allowed_tools=(QA_RETRIEVAL_TOOL_ID,),
                execution_mode="react",
            ),
            model=ModelSpec("model", "provider"),
            provider=ProviderSpec("provider", kind="test"),
            phase=ExecutionPhase.SINGLE,
            semantic_protocol=QA_VERIFIED_ANSWER_LINEAGE_PROTOCOL,
        )
        search = {
            "observation_status": "success",
            "executed_action": {
                "kind": "tool",
                "name": "search",
                "resource_id": QA_RETRIEVAL_TOOL_ID,
                "arguments": {"query": "Billboard first American hit chart", "limit": 5},
            },
            "result": {
                "operation": "search",
                "query": "Billboard first American hit chart",
                "top_k": 5,
                "passage_ids": ["p1"],
            },
        }
        read = {
            "observation_status": "success",
            "executed_action": {
                "kind": "tool",
                "name": "read",
                "resource_id": QA_RETRIEVAL_TOOL_ID,
                "arguments": {"passage_id": "p1"},
            },
            "result": {
                "operation": "read",
                "passage_id": "p1",
                "passage": {
                    "title": "Billboard",
                    "text": "Billboard is an American music magazine.",
                },
            },
        }
        structural_error = {
            "observation_status": "schema_invalid",
            "public_error_code": (
                "qa_semantic_artifact_invalid: Reasoner answer-bearing "
                "proposition has no deterministic entity binding"
            ),
        }

        first_state = adapter._required_evidence_state(
            request,
            [search, read, structural_error],
        )
        first_domain, first_completion = adapter._state_conditioned_action_domain(
            request,
            [search, read, structural_error],
        )
        self.assertEqual("structure", first_state.semantic_repair_kind)
        self.assertEqual(1, first_state.semantic_repair_attempt_count)
        self.assertEqual(frozenset(), first_domain)
        self.assertTrue(first_completion)

        observations = [search, read, structural_error, dict(structural_error)]
        state = adapter._required_evidence_state(request, observations)
        domain, completion = adapter._state_conditioned_action_domain(
            request,
            observations,
        )
        self.assertEqual("structure", state.semantic_repair_kind)
        self.assertEqual(2, state.semantic_repair_attempt_count)
        self.assertEqual(("p1",), state.read_passage_ids)
        self.assertEqual(frozenset(), domain)
        self.assertTrue(completion)

        visible = adapter._model_visible_observations(observations)
        self.assertEqual(2, visible[-1]["repeat_count"])
        self.assertNotIn("executed_action", visible[1])
        self.assertEqual("read", visible[1]["result"]["operation"])
        self.assertIn(
            "do not add a search or read",
            visible[-1]["repair_instruction"],
        )
        contract = adapter._contract(request, observations)
        self.assertIn("The required successful non-empty qa-retrieval reads are present", contract)
        self.assertIn('"semantic_repair_kind":"structure"', contract)
        self.assertIn('"semantic_repair_attempt_count":2', contract)

    async def test_duplicate_normalized_query_exposes_query_rewrite_feedback(
        self,
    ) -> None:
        visible = QARetrievalReactExecutionAdapter._model_visible_observations(
            [
                {
                    "observation_status": "schema_invalid",
                    "public_error_code": (
                        "qa_retrieval_duplicate_normalized_query"
                    ),
                }
            ]
        )

        self.assertEqual(1, len(visible))
        self.assertIn(
            "distinct normalized entity-and-relation query",
            visible[0]["repair_instruction"],
        )
        self.assertIn("strictly larger top_k", visible[0]["repair_instruction"])
        self.assertIn("(query, top_k) pair", visible[0]["repair_instruction"])
        self.assertIn(
            "does not advance retrieval-strategy progress",
            visible[0]["repair_instruction"],
        )

    def test_strategy_labels_are_not_executable_retrieval_queries(self) -> None:
        adapter = QARetrievalReactExecutionAdapter(
            gateway=SimpleNamespace(generate=lambda request: None),
            tool_registry=build_qa_tool_registry(FakeIndex()),
            max_turns=16,
            max_tool_calls=12,
            task_type="factual_qa",
            completion_policy="required_evidence",
        )
        question = (
            "In which decade did Billboard magazine first publish an American "
            "hit chart?"
        )
        request = AgentRequest(
            request_id="trivia:strategy-label-gate",
            run_id="trivia",
            graph_revision=1,
            problem=question,
            agent=AgentNode(
                "retriever",
                "model",
                "retrieve the original entity and relation",
                role_family="evidence_retriever",
                allowed_tools=(QA_RETRIEVAL_TOOL_ID,),
                execution_mode="react",
            ),
            model=ModelSpec("model", "provider"),
            provider=ProviderSpec("provider", kind="test"),
            phase=ExecutionPhase.SINGLE,
            semantic_protocol=QA_VERIFIED_ANSWER_LINEAGE_PROTOCOL,
        )
        for query in (
            "Billboard magazine first publish hit chart spelling normalization",
            "Billboard magazine first publish hit chart alias",
            "Billboard magazine first publish hit chart query rewriting",
        ):
            with self.subTest(query=query):
                action = StructuredAction(
                    ActionKind.TOOL,
                    "search",
                    {"query": query, "limit": 5},
                    resource_id=QA_RETRIEVAL_TOOL_ID,
                )
                self.assertEqual(
                    "qa_retrieval_query_strategy_label_injection",
                    adapter._tool_action_error(
                        request=request,
                        action=action,
                        observations=[],
                    ),
                )
        rejected_state = adapter._required_evidence_state(
            request,
            [
                {
                    "observation_status": "schema_invalid",
                    "public_error_code": (
                        "qa_retrieval_query_strategy_label_injection"
                    ),
                    "executed_action": {
                        "arguments": {
                            "query": (
                                "Billboard magazine first publish hit chart alias"
                            ),
                            "limit": 5,
                        },
                        "kind": "tool",
                        "name": "search",
                        "resource_id": QA_RETRIEVAL_TOOL_ID,
                        "skill_id": None,
                    },
                }
            ],
        )
        self.assertEqual(0, rejected_state.dispatched_tool_calls)
        self.assertEqual(0, rejected_state.strategy_progress_count)
        initial_contract = adapter._contract(request, [])
        self.assertIn("Current strategy semantics", initial_contract)
        self.assertNotIn("alias_expansion", initial_contract)
        neutral = StructuredAction(
            ActionKind.TOOL,
            "search",
            {
                "query": (
                    "Billboard magazine first publish music hit parade"
                ),
                "limit": 5,
            },
            resource_id=QA_RETRIEVAL_TOOL_ID,
        )
        self.assertIsNone(
            adapter._tool_action_error(
                request=request,
                action=neutral,
                observations=[],
            )
        )
        question_with_legitimate_alias_term = replace(
            request,
            problem="What alias did the performer use?",
        )
        legitimate = StructuredAction(
            ActionKind.TOOL,
            "search",
            {"query": "performer alias stage name", "limit": 5},
            resource_id=QA_RETRIEVAL_TOOL_ID,
        )
        self.assertIsNone(
            adapter._tool_action_error(
                request=question_with_legitimate_alias_term,
                action=legitimate,
                observations=[],
            )
        )
        visible = adapter._model_visible_observations(
            [
                {
                    "observation_status": "schema_invalid",
                    "public_error_code": (
                        "qa_retrieval_query_strategy_label_injection"
                    ),
                }
            ]
        )
        self.assertIn(
            "never copy orchestration labels",
            visible[0]["repair_instruction"],
        )

    async def test_search_receipt_preserves_frozen_identity_query_top_k_and_ids(self) -> None:
        index = FakeIndex()
        registry = build_qa_tool_registry(index)

        result, receipt = await registry.ainvoke_with_receipt(
            QA_RETRIEVAL_TOOL_ID,
            ToolRequest("search", {"query": "Ada Lovelace", "limit": 3}),
        )

        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(["p1"], result.value["passage_ids"])
        self.assertEqual("Ada Lovelace", result.value["query"])
        self.assertEqual(3, result.value["top_k"])
        self.assertEqual(
            {
                "source": "public-wikipedia",
                "corpus_version": "snapshot-2026-08",
                "index_id": "frozen-index-v1",
                "index_format": "skillev-public-retrieval-index@2",
                "retrieval_backend": "sqlite-fts5-lexical",
            },
            result.value["retrieval_index"],
        )
        self.assertEqual([("Ada Lovelace", 3)], index.search_calls)
        self.assertEqual("frozen-index-v1", receipt.tool_version)
        self.assertGreaterEqual(receipt.latency_ms, 0.0)
        self.assertEqual(
            {"query": "Ada Lovelace", "limit": 3},
            dict(receipt.request.arguments),
        )

    async def test_read_receipt_contains_public_passage_and_index_version(self) -> None:
        index = FakeIndex()
        registry = build_qa_tool_registry(index)

        result, receipt = await registry.ainvoke_with_receipt(
            QA_RETRIEVAL_TOOL_ID,
            ToolRequest("read", {"passage_id": "p1"}),
        )

        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual("Full public text.", result.value["passage"]["text"])
        self.assertEqual("frozen-index-v1", result.value["retrieval_index"]["index_id"])
        self.assertEqual(["p1"], index.read_calls)
        self.assertGreaterEqual(receipt.latency_ms, 0.0)

    async def test_tool_schemas_reject_evaluator_or_answer_key_arguments(self) -> None:
        index = FakeIndex()
        registry = build_qa_tool_registry(index)

        result, receipt = await registry.ainvoke_with_receipt(
            QA_RETRIEVAL_TOOL_ID,
            ToolRequest(
                "search",
                {
                    "query": "Ada Lovelace",
                    "limit": 1,
                    "accepted_answers": ["evaluator-only"],
                },
            ),
        )

        self.assertIsNone(result)
        self.assertEqual("ValueError", receipt.error_type)
        self.assertEqual([], index.search_calls)

    def test_registry_exposes_one_skillflow_search_read_resource(self) -> None:
        registry = build_qa_tool_registry(FakeIndex())

        self.assertEqual((QA_RETRIEVAL_TOOL_ID,), registry.resource_ids)
        retrieval = registry.require_capability(QA_RETRIEVAL_TOOL_ID)
        self.assertEqual(("hotpotqa", "triviaqa"), retrieval.dataset_scope)
        self.assertEqual(("read", "search"), retrieval.action_names)
        self.assertEqual(
            ["query", "limit"],
            retrieval.action_schemas["search"]["required"],
        )
        self.assertEqual(
            ["passage_id"],
            retrieval.action_schemas["read"]["required"],
        )
        self.assertIn(
            "successful search",
            retrieval.action_schemas["read"]["properties"]["passage_id"][
                "description"
            ],
        )
        self.assertEqual("frozen-index-v1", retrieval.version)
        self.assertEqual("none", retrieval.side_effect)

    def test_open_factory_uses_skillflow_open_and_owns_close(self) -> None:
        index = FakeIndex()

        class FakeRetrievalIndexClass:
            opened_path: Path | None = None

            @classmethod
            def open(cls, path: Path) -> FakeIndex:
                cls.opened_path = path
                return index

        with patch(
            "src.interactive.qa_tool_adapter._load_retrieval_index_class",
            return_value=FakeRetrievalIndexClass,
        ):
            with open_qa_tool_registry(
                index_path="/tmp/frozen-public-index.sqlite3",
                skillflow_source="/tmp/skillflow-source",
            ) as opened:
                self.assertEqual(
                    "frozen-index-v1",
                    opened.retrieval_index_identity["index_id"],
                )
                self.assertIn(
                    QA_RETRIEVAL_TOOL_ID, opened.registry.resource_ids
                )
                self.assertFalse(index.closed)

        self.assertEqual(
            Path("/tmp/frozen-public-index.sqlite3"),
            FakeRetrievalIndexClass.opened_path,
        )
        self.assertTrue(index.closed)

    async def test_open_factory_interrupts_inflight_sqlite_search_on_timeout(
        self,
    ) -> None:
        class ProgressConnection:
            def __init__(self) -> None:
                self.handler: object = None

            def set_progress_handler(
                self,
                handler: object,
                instruction_interval: int,
            ) -> None:
                del instruction_interval
                self.handler = handler

        class SlowIndex(FakeIndex):
            def __init__(self) -> None:
                super().__init__()
                self._connection = ProgressConnection()
                self.started = Event()
                self.interrupted = Event()

            def search(self, query: str, *, limit: int) -> tuple[FakeHit, ...]:
                del query, limit
                self.started.set()
                while True:
                    handler = self._connection.handler
                    if callable(handler) and handler():
                        self.interrupted.set()
                        raise RuntimeError("sqlite query interrupted")
                    time.sleep(0.001)

        index = SlowIndex()

        class SlowRetrievalIndexClass:
            @classmethod
            def open(cls, path: Path) -> SlowIndex:
                del path
                return index

        with patch(
            "src.interactive.qa_tool_adapter._load_retrieval_index_class",
            return_value=SlowRetrievalIndexClass,
        ):
            with open_qa_tool_registry(
                index_path="/tmp/frozen-public-index.sqlite3",
                skillflow_source="/tmp/skillflow-source",
                timeout_seconds=0.02,
            ) as opened:
                result, receipt = await opened.registry.ainvoke_with_receipt(
                    QA_RETRIEVAL_TOOL_ID,
                    ToolRequest(
                        "search",
                        {"query": "slow public query", "limit": 1},
                    ),
                )

                self.assertIsNone(result)
                self.assertEqual("TimeoutError", receipt.error_type)
                self.assertTrue(index.started.is_set())
                self.assertTrue(index.interrupted.is_set())

        self.assertTrue(index.closed)

    def test_provided_context_factory_reuses_skillflow_builder_and_removes_temp_index(self) -> None:
        index = FakeIndex()
        captured: dict[str, object] = {}

        class FakeDocumentPassage:
            def __init__(self, **kwargs: object) -> None:
                self.__dict__.update(kwargs)

        class FakeRetrievalIndexClass:
            @classmethod
            def open(cls, path: Path) -> FakeIndex:
                captured["opened_path"] = path
                return index

        def build(path: Path, passages: tuple[object, ...], **kwargs: object) -> None:
            captured["built_path"] = path
            captured["passages"] = passages
            captured["identity"] = kwargs
            path.write_text("fixture", encoding="utf-8")

        module = SimpleNamespace(
            DocumentPassage=FakeDocumentPassage,
            RetrievalIndex=FakeRetrievalIndexClass,
            build_retrieval_index=build,
        )
        with patch(
            "src.interactive.qa_tool_adapter._load_retrieval_module",
            return_value=module,
        ):
            opened = open_provided_context_qa_tool_registry(
                ["[Widsith] Widsith preserves a list of kings."],
                skillflow_source="/tmp/skillflow-source",
            )
            built_path = captured["built_path"]
            assert isinstance(built_path, Path)
            self.assertTrue(built_path.is_file())
            supplied = captured["passages"]
            assert isinstance(supplied, tuple)
            self.assertEqual("Widsith", supplied[0].title)
            self.assertEqual(
                "Widsith preserves a list of kings.", supplied[0].text
            )
            self.assertEqual(
                "benchmark-provided-context",
                captured["identity"]["corpus_name"],
            )
            opened.close()

        self.assertTrue(index.closed)
        self.assertFalse(built_path.parent.exists())

    def test_multi_hop_adapter_injects_upstream_skillflow_guidance(self) -> None:
        request = AgentRequest(
            request_id="qa:contract",
            run_id="qa",
            graph_revision=1,
            problem="What book contains Widsith?",
            agent=AgentNode(
                "retriever",
                "model",
                "find and chain the relevant passages",
                allowed_tools=(QA_RETRIEVAL_TOOL_ID,),
                execution_mode="react",
            ),
            model=ModelSpec("model", "provider"),
            provider=ProviderSpec("provider", kind="test"),
            phase=ExecutionPhase.SINGLE,
        )
        adapter = QARetrievalReactExecutionAdapter(
            gateway=SimpleNamespace(generate=lambda request: None),
            tool_registry=build_qa_tool_registry(FakeIndex()),
            max_turns=3,
            max_tool_calls=2,
            task_type="multi_hop_qa",
        )

        rendered = adapter._contract(request, [])

        self.assertIn("Chain evidence across passages", rendered)
        self.assertIn("specific entity names (not the full question)", rendered)
        self.assertNotIn("ReAct only as the execution schedule", rendered)
        self.assertNotIn("subject/entity, predicate/relation", rendered)
        self.assertNotIn("unexpectedly equal", rendered)
        self.assertNotIn('"value":"final artifact"', rendered)
        self.assertIn(
            "The completed artifact required by the Agent contract",
            rendered,
        )

        format_predecessor = AgentRequest(
            request_id=request.request_id,
            run_id=request.run_id,
            graph_revision=request.graph_revision,
            problem=request.problem,
            agent=request.agent,
            model=request.model,
            provider=request.provider,
            phase=request.phase,
            is_format_predecessor=True,
        )
        rendered_predecessor = adapter._contract(format_predecessor, [])
        self.assertIn("Candidate answer:", rendered_predecessor)
        self.assertIn("Evidence:", rendered_predecessor)

    def test_reasoner_and_verifier_receive_distinct_completion_guidance(self) -> None:
        adapter = QARetrievalReactExecutionAdapter(
            gateway=SimpleNamespace(generate=lambda request: None),
            tool_registry=build_qa_tool_registry(FakeIndex()),
            max_turns=3,
            max_tool_calls=2,
            task_type="multi_hop_qa",
        )

        def role_request(role_family: str) -> AgentRequest:
            return AgentRequest(
                request_id=f"qa:{role_family}",
                run_id="qa",
                graph_revision=1,
                problem="Which entity has the larger value?",
                agent=AgentNode(
                    role_family,
                    "model",
                    "preserve the original question scope",
                    role_family=role_family,
                    allowed_tools=(QA_RETRIEVAL_TOOL_ID,),
                    execution_mode="react",
                ),
                model=ModelSpec("model", "provider"),
                provider=ProviderSpec("provider", kind="test"),
                phase=ExecutionPhase.SINGLE,
                is_format_predecessor=role_family == "verifier",
                semantic_protocol="hotpotqa_verified_answer_slot_v1",
            )

        reasoner = adapter._contract(role_request("reasoner"), [])
        verifier = adapter._contract(role_request("verifier"), [])

        self.assertIn("evidence_propositions", reasoner)
        self.assertIn("bind the candidate", reasoner)
        self.assertIn(
            "minimal but complete evidence-aligned referential surface",
            reasoner,
        )
        self.assertIn("possessive marker", reasoner)
        self.assertIn("complete possessor entity mention", reasoner)
        self.assertIn("Do not replace the Reasoner's candidate", verifier)
        self.assertIn("seven explicit boolean check fields", verifier)
        self.assertIn("Set supported only when all checks pass", verifier)
        self.assertIn(
            "minimal but complete evidence-aligned referential surface",
            verifier,
        )
        self.assertIn("drops part of the possessor entity mention", verifier)
        self.assertNotIn("bind the candidate", verifier)

        completion_schema = adapter._completion_arguments_schema(
            role_request("reasoner")
        )
        semantic_properties = completion_schema["properties"]["value"]["properties"]
        proposition_properties = semantic_properties["evidence_propositions"][
            "items"
        ]["properties"]
        for field in ("subject", "object_or_attribute_value"):
            description = proposition_properties[field]["description"]
            self.assertIn(
                "minimal but complete evidence-aligned referential surface",
                description,
            )
            self.assertIn("complete possessor mention", description)
            self.assertIn("possessive marker", description)
            self.assertIn("name suffix", description)
        candidate_description = semantic_properties["candidate_answer"][
            "description"
        ]
        self.assertIn(
            "minimal but complete evidence-aligned referential surface",
            candidate_description,
        )
        self.assertIn("not a strict subspan", candidate_description)

        default_reasoner = replace(
            role_request("reasoner"),
            semantic_protocol="none",
        )
        default_contract = adapter._contract(default_reasoner, [])
        self.assertNotIn("evidence_propositions", default_contract)
        self.assertNotIn("unexpectedly equal", default_contract)

    def test_required_evidence_shows_reasoner_artifact_only_at_completion_state(
        self,
    ) -> None:
        adapter = QARetrievalReactExecutionAdapter(
            gateway=SimpleNamespace(generate=lambda request: None),
            tool_registry=build_qa_tool_registry(FakeIndex()),
            max_turns=4,
            max_tool_calls=2,
            task_type="multi_hop_qa",
            completion_policy="required_evidence",
        )
        request = AgentRequest(
            request_id="qa:semantic-stage",
            run_id="qa",
            graph_revision=1,
            problem="Which city is requested?",
            agent=AgentNode(
                "reasoner",
                "model",
                "preserve scope and determine the semantic answer",
                role_family="reasoner",
                allowed_tools=(QA_RETRIEVAL_TOOL_ID,),
                execution_mode="react",
            ),
            model=ModelSpec("model", "provider"),
            provider=ProviderSpec("provider", kind="test"),
            phase=ExecutionPhase.SINGLE,
            semantic_protocol="hotpotqa_verified_answer_slot_v1",
        )
        search_observation = {
            "observation_status": "success",
            "result": {
                "operation": "search",
                "passage_ids": ["p1"],
            },
        }
        read_observation = {
            "observation_status": "success",
            "result": {
                "operation": "read",
                "passage_id": "p1",
                "passage": {"text": "Paris is the capital of France."},
            },
        }

        search_contract = adapter._contract(request, [])
        read_contract = adapter._contract(request, [search_observation])
        completion_contract = adapter._contract(
            request,
            [search_observation, read_observation],
        )

        self.assertNotIn("six structured fields", search_contract)
        self.assertNotIn("six structured fields", read_contract)
        self.assertIn("six structured fields", completion_contract)
        self.assertNotIn('"arguments":{"type":"object"', search_contract)
        self.assertNotIn('"arguments":{"type":"object"', read_contract)
        self.assertIn("Completion is not admitted in this Tool-only state", search_contract)
        self.assertIn("Completion is not admitted in this Tool-only state", read_contract)

    def test_unified_factual_schema_uses_one_grounded_proposition(self) -> None:
        adapter = QARetrievalReactExecutionAdapter(
            gateway=SimpleNamespace(generate=lambda request: None),
            tool_registry=build_qa_tool_registry(FakeIndex()),
            max_turns=6,
            max_tool_calls=10,
            task_type="factual_qa",
            completion_policy="required_evidence",
        )
        request = AgentRequest(
            request_id="trivia:semantic-schema",
            run_id="trivia",
            graph_revision=1,
            problem="Where was the historical figure born?",
            agent=AgentNode(
                "reasoner",
                "model",
                "retrieve evidence and bind the requested relation",
                role_family="reasoner",
                allowed_tools=(QA_RETRIEVAL_TOOL_ID,),
                execution_mode="react",
            ),
            model=ModelSpec("model", "provider"),
            provider=ProviderSpec("provider", kind="test"),
            phase=ExecutionPhase.SINGLE,
            semantic_protocol=QA_VERIFIED_ANSWER_LINEAGE_PROTOCOL,
        )

        schema = adapter._completion_arguments_schema(request)
        fields = schema["properties"]["value"]["properties"]
        self.assertEqual(
            "Where was the historical figure born?",
            fields["question_scope"]["const"],
        )
        self.assertEqual(
            "location",
            fields["answer_slot"]["properties"]["answer_type"]["const"],
        )
        self.assertEqual(1, fields["evidence_propositions"]["minItems"])
        self.assertEqual(1, fields["multi_hop_chain"]["minItems"])
        self.assertIn(
            "explicit identity binding",
            fields["evidence_propositions"]["items"]["properties"]
            ["subject"]["description"],
        )

        initial_schema = adapter._state_conditioned_response_schema(request, [])
        assert initial_schema is not None
        self.assertEqual(
            5,
            initial_schema["properties"]["arguments"]["properties"]
            ["limit"]["const"],
        )
        initial_contract = adapter._contract(request, [])
        self.assertIn("`initial_retrieval`", initial_contract)
        self.assertIn("entity identity and target relation", initial_contract)

    def test_location_containment_lineage_is_state_conditioned(self) -> None:
        question = "Where in Arcadia was Professor Mira Hale born?"
        first_span = (
            "Professor Mira Hale was born in East Ward, North Province."
        )
        first_hop = {
            "subject": "Professor Mira Hale",
            "relation": "was born in",
            "object_or_attribute_value": "East Ward",
            "qualifiers": [],
            "evidence_span": first_span,
        }

        def fields(
            *,
            answer_type: str = "location",
            propositions: list[dict[str, object]] | None = None,
            proposition_index: int = 0,
            candidate: str = "East Ward",
            chain: list[str] | None = None,
        ) -> dict[str, object]:
            return {
                "answer_slot": {
                    "answer_type": answer_type,
                    "answer_cardinality": "single",
                    "qualifiers": ["in Arcadia"],
                    "proposition_index": proposition_index,
                    "answer_field": "object_or_attribute_value",
                },
                "evidence_propositions": propositions or [first_hop],
                "multi_hop_chain": chain or [
                    "Professor Mira Hale --was born in--> East Ward"
                ],
                "candidate_answer": candidate,
            }

        self.assertEqual(
            "qa_location_containment_lineage_missing",
            _location_containment_lineage_issue(
                original_question=question,
                reasoner_fields=fields(),
            ),
        )

        distractor_scope_first_hop = dict(first_hop)
        distractor_scope_first_hop["evidence_span"] = (
            first_span
            + " Her father later moved to Arcadia and worked there."
        )
        self.assertEqual(
            "qa_location_containment_lineage_missing",
            _location_containment_lineage_issue(
                original_question=question,
                reasoner_fields=fields(
                    propositions=[distractor_scope_first_hop],
                ),
            ),
        )

        containment_span = (
            "East Ward is a district and part of the city of Riverton in Arcadia."
        )
        containment = {
            "subject": "East Ward",
            "relation": "is part of the city of",
            "object_or_attribute_value": "Riverton",
            "qualifiers": ["in Arcadia"],
            "evidence_span": containment_span,
        }
        valid_fields = fields(
            propositions=[first_hop, containment],
            proposition_index=1,
            candidate="Riverton",
            chain=[
                "Professor Mira Hale --was born in--> East Ward",
                "East Ward --part of the city of--> Riverton",
            ],
        )
        self.assertIsNone(
            _location_containment_lineage_issue(
                original_question=question,
                reasoner_fields=valid_fields,
                read_evidence_texts=(
                    FakeReadReceiptText(
                        containment_span,
                        passage_title="East Ward",
                    ),
                ),
            )
        )
        missing_scope = json.loads(json.dumps(valid_fields))
        missing_scope["evidence_propositions"][1]["evidence_span"] = (
            "East Ward is a district and part of the city of Riverton."
        )
        self.assertEqual(
            "qa_location_containment_lineage_missing",
            _location_containment_lineage_issue(
                original_question=question,
                reasoner_fields=missing_scope,
                read_evidence_texts=(
                    FakeReadReceiptText(
                        missing_scope["evidence_propositions"][1][
                            "evidence_span"
                        ],
                        passage_title="East Ward",
                    ),
                ),
            ),
        )
        broken_bridge = json.loads(json.dumps(valid_fields))
        broken_bridge["evidence_propositions"][1]["subject"] = "West Ward"
        broken_bridge["evidence_propositions"][1]["evidence_span"] = (
            "West Ward is a district and part of the city of Riverton in Arcadia."
        )
        self.assertEqual(
            "qa_location_containment_lineage_missing",
            _location_containment_lineage_issue(
                original_question=question,
                reasoner_fields=broken_bridge,
                read_evidence_texts=(
                    FakeReadReceiptText(
                        broken_bridge["evidence_propositions"][1][
                            "evidence_span"
                        ],
                        passage_title="West Ward",
                    ),
                ),
            ),
        )
        untyped_child = json.loads(json.dumps(valid_fields))
        untyped_child["evidence_propositions"][1]["evidence_span"] = (
            "East Ward is near the city of Riverton in Arcadia."
        )
        self.assertEqual(
            "qa_location_containment_lineage_missing",
            _location_containment_lineage_issue(
                original_question=question,
                reasoner_fields=untyped_child,
                read_evidence_texts=(
                    FakeReadReceiptText(
                        untyped_child["evidence_propositions"][1][
                            "evidence_span"
                        ],
                        passage_title="East Ward",
                    ),
                ),
            ),
        )

        scope_grounded_first_hop = dict(first_hop)
        scope_grounded_first_hop["evidence_span"] = (
            "Professor Mira Hale was born in East Ward, Arcadia."
        )
        self.assertIsNone(
            _location_containment_lineage_issue(
                original_question=question,
                reasoner_fields=fields(
                    propositions=[scope_grounded_first_hop],
                ),
            )
        )
        self.assertIsNone(
            _location_containment_lineage_issue(
                original_question="Where was Professor Mira Hale born?",
                reasoner_fields=fields(),
            )
        )
        self.assertIsNone(
            _location_containment_lineage_issue(
                original_question=question,
                reasoner_fields=fields(answer_type="entity"),
            )
        )

        country_span = "East Ward belongs to the country of Arcadia."
        country_promotion = dict(containment)
        country_promotion.update(
            relation="belongs to the country of",
            object_or_attribute_value="Arcadia",
            evidence_span=country_span,
        )
        self.assertEqual(
            "qa_location_containment_lineage_missing",
            _location_containment_lineage_issue(
                original_question=question,
                reasoner_fields=fields(
                    propositions=[first_hop, country_promotion],
                    proposition_index=1,
                    candidate="Arcadia",
                    chain=[
                        "Professor Mira Hale --was born in--> East Ward",
                        "East Ward --belongs to the country of--> Arcadia",
                    ],
                ),
                read_evidence_texts=(
                    FakeReadReceiptText(
                        country_span,
                        passage_title="East Ward",
                    ),
                ),
            ),
        )

    def test_comma_qualified_location_resolution_uses_public_type_lineage(
        self,
    ) -> None:
        question = "Where in Arcadia was Dr Rowan Vale born?"
        first_span = (
            "Dr Rowan Vale was born in Harbor Quarter, North Province."
        )
        first_hop = {
            "subject": "Dr Rowan Vale",
            "relation": "was born in",
            "object_or_attribute_value": "Harbor Quarter, North Province",
            "qualifiers": [],
            "evidence_span": first_span,
        }

        def fields(
            *,
            propositions: list[dict[str, object]],
            proposition_index: int,
            answer_field: str,
            candidate: str,
            chain: list[str],
        ) -> dict[str, object]:
            return {
                "answer_slot": {
                    "answer_type": "location",
                    "answer_cardinality": "single",
                    "qualifiers": ["in Arcadia"],
                    "proposition_index": proposition_index,
                    "answer_field": answer_field,
                },
                "evidence_propositions": propositions,
                "multi_hop_chain": chain,
                "candidate_answer": candidate,
            }

        unresolved = fields(
            propositions=[first_hop],
            proposition_index=0,
            answer_field="object_or_attribute_value",
            candidate="Harbor Quarter, North Province",
            chain=[
                "Dr Rowan Vale --was born in--> Harbor Quarter, North Province"
            ],
        )
        self.assertEqual(
            "qa_location_containment_lineage_missing",
            _location_containment_lineage_issue(
                original_question=question,
                reasoner_fields=unresolved,
                read_evidence_texts=(
                    FakeReadReceiptText(
                        first_span,
                        passage_title="Rowan Vale",
                    ),
                ),
            ),
        )

        containment_span = (
            "Harbor Quarter is part of the city of Riverton in Arcadia."
        )
        containment = {
            "subject": "Harbor Quarter",
            "relation": "is part of the city of",
            "object_or_attribute_value": "Riverton",
            "qualifiers": ["in Arcadia"],
            "evidence_span": containment_span,
        }
        containment_fields = fields(
            propositions=[first_hop, containment],
            proposition_index=1,
            answer_field="object_or_attribute_value",
            candidate="Riverton",
            chain=[
                "Dr Rowan Vale --was born in--> Harbor Quarter, North Province",
                "Harbor Quarter --part of the city of--> Riverton",
            ],
        )
        containment_read = FakeReadReceiptText(
            containment_span,
            passage_title="Harbor Quarter, Riverton",
        )
        self.assertIsNone(
            _location_containment_lineage_issue(
                original_question=question,
                reasoner_fields=containment_fields,
                read_evidence_texts=(containment_read,),
            )
        )
        leading_component_chain = json.loads(json.dumps(containment_fields))
        leading_component_chain["multi_hop_chain"][0] = (
            "Dr Rowan Vale --was born in--> Harbor Quarter"
        )
        self.assertIsNone(
            _location_containment_lineage_issue(
                original_question=question,
                reasoner_fields=leading_component_chain,
                read_evidence_texts=(containment_read,),
            )
        )
        reordered_resolution = json.loads(json.dumps(leading_component_chain))
        reordered_resolution["evidence_propositions"] = [
            reordered_resolution["evidence_propositions"][1],
            reordered_resolution["evidence_propositions"][0],
        ]
        reordered_resolution["answer_slot"]["proposition_index"] = 0
        self.assertIsNone(
            _location_containment_lineage_issue(
                original_question=question,
                reasoner_fields=reordered_resolution,
                read_evidence_texts=(containment_read,),
            )
        )
        self.assertEqual(
            "qa_location_containment_lineage_missing",
            _location_containment_lineage_issue(
                original_question=question,
                reasoner_fields=containment_fields,
            ),
        )

        prefix_subject = json.loads(json.dumps(containment_fields))
        prefix_subject["evidence_propositions"][1]["subject"] = "Harbor"
        prefix_subject["evidence_propositions"][1]["evidence_span"] = (
            "Harbor is part of the city of Riverton in Arcadia."
        )
        self.assertEqual(
            "qa_location_containment_lineage_missing",
            _location_containment_lineage_issue(
                original_question=question,
                reasoner_fields=prefix_subject,
                read_evidence_texts=(
                    FakeReadReceiptText(
                        prefix_subject["evidence_propositions"][1][
                            "evidence_span"
                        ],
                        passage_title="Harbor, Riverton",
                    ),
                ),
            ),
        )
        self.assertEqual(
            "qa_location_containment_lineage_missing",
            _location_containment_lineage_issue(
                original_question=question,
                reasoner_fields=containment_fields,
                read_evidence_texts=(
                    FakeReadReceiptText(
                        containment_span,
                        passage_title="Other Quarter, Riverton",
                    ),
                ),
            ),
        )

        settlement_first_span = (
            "Dr Rowan Vale was born in Riverton, North Province."
        )
        settlement_first_hop = {
            "subject": "Dr Rowan Vale",
            "relation": "was born in",
            "object_or_attribute_value": "Riverton, North Province",
            "qualifiers": [],
            "evidence_span": settlement_first_span,
        }
        settlement_span = "Riverton is a city in Arcadia."
        settlement = {
            "subject": "Riverton",
            "relation": "is a city in",
            "object_or_attribute_value": "Arcadia",
            "qualifiers": ["in Arcadia"],
            "evidence_span": settlement_span,
        }
        settlement_fields = fields(
            propositions=[settlement_first_hop, settlement],
            proposition_index=1,
            answer_field="subject",
            candidate="Riverton",
            chain=[
                "Dr Rowan Vale --was born in--> Riverton, North Province",
                "Riverton --is a city in--> Arcadia",
            ],
        )
        self.assertIsNone(
            _location_containment_lineage_issue(
                original_question=question,
                reasoner_fields=settlement_fields,
                read_evidence_texts=(
                    FakeReadReceiptText(
                        settlement_span,
                        passage_title="Riverton",
                    ),
                ),
            )
        )

        title_only = json.loads(json.dumps(settlement_fields))
        title_only["evidence_propositions"][1]["relation"] = "is located in"
        title_only["evidence_propositions"][1]["evidence_span"] = (
            "Riverton is located in Arcadia."
        )
        self.assertEqual(
            "qa_location_containment_lineage_missing",
            _location_containment_lineage_issue(
                original_question=question,
                reasoner_fields=title_only,
                read_evidence_texts=(
                    FakeReadReceiptText(
                        title_only["evidence_propositions"][1][
                            "evidence_span"
                        ],
                        passage_title="Riverton",
                    ),
                ),
            ),
        )

        district_as_answer = json.loads(json.dumps(containment_fields))
        district_as_answer["answer_slot"]["answer_field"] = "subject"
        district_as_answer["candidate_answer"] = "Harbor Quarter"
        self.assertEqual(
            "qa_location_containment_lineage_missing",
            _location_containment_lineage_issue(
                original_question=question,
                reasoner_fields=district_as_answer,
                read_evidence_texts=(containment_read,),
            ),
        )

        scope_grounded_span = "Dr Rowan Vale was born in Riverton, Arcadia."
        scope_grounded = fields(
            propositions=[
                {
                    **settlement_first_hop,
                    "object_or_attribute_value": "Riverton, Arcadia",
                    "evidence_span": scope_grounded_span,
                }
            ],
            proposition_index=0,
            answer_field="object_or_attribute_value",
            candidate="Riverton, Arcadia",
            chain=["Dr Rowan Vale --was born in--> Riverton, Arcadia"],
        )
        self.assertIsNone(
            _location_containment_lineage_issue(
                original_question=question,
                reasoner_fields=scope_grounded,
                read_evidence_texts=(
                    FakeReadReceiptText(
                        scope_grounded_span,
                        passage_title="Rowan Vale",
                    ),
                ),
            )
        )

    def test_location_containment_repair_feedback_is_model_visible_without_leakage(
        self,
    ) -> None:
        observations = [
            {
                "observation_status": "schema_invalid",
                "public_error_code": (
                    "qa_semantic_evidence_provenance_invalid: "
                    "qa_location_containment_lineage_missing"
                ),
                "executed_action": {
                    "kind": "complete",
                    "name": "complete",
                    "arguments": {
                        "value": {
                            "candidate_answer": "Unverified Locality",
                            "accepted_answers": ["Hidden Parent"],
                        }
                    },
                },
            }
        ]

        visible = QARetrievalReactExecutionAdapter._model_visible_observations(
            observations
        )

        rendered = json.dumps(visible, ensure_ascii=False)
        self.assertIn("Preserve the receipt-grounded first-hop", rendered)
        self.assertIn("search/read continuation", rendered)
        self.assertIn("identity and geographic type", rendered)
        self.assertIn("preserve it", rendered)
        self.assertIn("passage title", rendered)
        self.assertIn("cannot supply type, containment, or scope", rendered)
        self.assertIn("do not guess a parent", rendered)
        self.assertIn("two distinct evidence_propositions", rendered)
        self.assertIn("Set answer_slot", rendered)
        self.assertNotIn("Unverified Locality", rendered)
        self.assertNotIn("Hidden Parent", rendered)
        self.assertNotIn("accepted_answers", rendered)
        self.assertNotIn("ground_truth", rendered)
        self.assertNotIn("evaluator", rendered)

    @staticmethod
    def _synthetic_location_repair_case(
        *,
        ordinary_strategy_count: int = 1,
        max_tool_calls: int = 12,
        max_turns: int = 20,
    ) -> tuple[
        QARetrievalReactExecutionAdapter,
        AgentRequest,
        list[dict[str, object]],
        dict[str, object],
    ]:
        """Build a public, answer-free typed location-repair state."""

        question = "Where in Arcadia was Professor Mira Hale born?"
        first_span = (
            "Professor Mira Hale was born in East Ward, North Province."
        )
        first_hop_artifact: dict[str, object] = {
            "question_scope": question,
            "answer_slot": {
                "answer_type": "location",
                "answer_cardinality": "single",
                "qualifiers": ["in Arcadia"],
                "proposition_index": 0,
                "answer_field": "object_or_attribute_value",
            },
            "evidence_propositions": [
                {
                    "subject": "Professor Mira Hale",
                    "relation": "was born in",
                    "object_or_attribute_value": (
                        "East Ward, North Province"
                    ),
                    "qualifiers": [],
                    "evidence_span": first_span,
                }
            ],
            "multi_hop_chain": [
                "Professor Mira Hale --was born in--> "
                "East Ward, North Province"
            ],
            "candidate_answer": "East Ward, North Province",
            "evidence": [first_span],
        }
        request = AgentRequest(
            request_id="trivia:synthetic-location-repair",
            run_id="trivia",
            graph_revision=1,
            problem=question,
            agent=AgentNode(
                "reasoner",
                "model",
                "retrieve evidence and bind the requested location relation",
                role_family="reasoner",
                allowed_tools=(QA_RETRIEVAL_TOOL_ID,),
                execution_mode="react",
            ),
            model=ModelSpec("model", "provider"),
            provider=ProviderSpec("provider", kind="test"),
            phase=ExecutionPhase.SINGLE,
            semantic_protocol=QA_VERIFIED_ANSWER_LINEAGE_PROTOCOL,
        )
        adapter = QARetrievalReactExecutionAdapter(
            gateway=SimpleNamespace(generate=lambda request: None),
            tool_registry=build_qa_tool_registry(FakeIndex()),
            max_turns=max_turns,
            max_tool_calls=max_tool_calls,
            task_type="factual_qa",
            completion_policy="required_evidence",
        )
        ordinary_queries = (
            "Professor Mira Hale birth place Arcadia",
            "Professor Mira Hale birthplace Arcadia",
            "Professor Mira Hale born location Arcadia",
            "Professor Mira Hale place of birth Arcadia",
            "Professor Mira Hale born Arcadia history",
        )
        observations: list[dict[str, object]] = []
        for index, query in enumerate(
            ordinary_queries[:ordinary_strategy_count],
            start=1,
        ):
            passage_id = f"first-hop-{index}"
            observations.extend(
                (
                    {
                        "observation_status": "success",
                        "executed_action": {
                            "kind": "tool",
                            "name": "search",
                            "resource_id": QA_RETRIEVAL_TOOL_ID,
                            "arguments": {
                                "query": query,
                                "limit": (5, 10, 15, 20, 25)[index - 1],
                            },
                        },
                        "result": {
                            "operation": "search",
                            "query": query,
                            "top_k": (5, 10, 15, 20, 25)[index - 1],
                            "passage_ids": [passage_id],
                            "hits": [
                                {
                                    "passage_id": passage_id,
                                    "title": "Professor Mira Hale",
                                    "snippet": first_span,
                                    "rank": 1,
                                }
                            ],
                        },
                    },
                    {
                        "observation_status": "success",
                        "executed_action": {
                            "kind": "tool",
                            "name": "read",
                            "resource_id": QA_RETRIEVAL_TOOL_ID,
                            "arguments": {"passage_id": passage_id},
                        },
                        "result": {
                            "operation": "read",
                            "passage_id": passage_id,
                            "passage": {
                                "title": "Professor Mira Hale",
                                "text": first_span,
                            },
                        },
                    },
                )
            )
        observations.append(
            {
                "observation_status": "schema_invalid",
                "public_error_code": (
                    "qa_semantic_evidence_provenance_invalid: "
                    "qa_location_containment_lineage_missing"
                ),
                "executed_action": {
                    "kind": "complete",
                    "name": "complete",
                    "resource_id": None,
                    "skill_id": None,
                    "arguments": {"value": first_hop_artifact},
                },
            }
        )
        return adapter, request, observations, first_hop_artifact

    @staticmethod
    def _append_irrelevant_location_relation_search_and_read(
        observations: list[dict[str, object]],
        *,
        query: str,
        limit: int,
        passage_id: str = "unrelated-location-result",
    ) -> None:
        """Append a successful Tool receipt that does not ground containment."""

        unrelated_span = (
            "The East Ward Gazette publishes a yearly arts calendar."
        )
        observations.extend(
            (
                {
                    "observation_status": "success",
                    "executed_action": {
                        "kind": "tool",
                        "name": "search",
                        "resource_id": QA_RETRIEVAL_TOOL_ID,
                        "arguments": {"query": query, "limit": limit},
                    },
                    "result": {
                        "operation": "search",
                        "query": query,
                        "top_k": limit,
                        "passage_ids": [passage_id],
                        "hits": [
                            {
                                "passage_id": passage_id,
                                "title": "East Ward Gazette",
                                "snippet": unrelated_span,
                                "rank": 1,
                            }
                        ],
                    },
                },
                {
                    "observation_status": "success",
                    "executed_action": {
                        "kind": "tool",
                        "name": "read",
                        "resource_id": QA_RETRIEVAL_TOOL_ID,
                        "arguments": {"passage_id": passage_id},
                    },
                    "result": {
                        "operation": "read",
                        "passage_id": passage_id,
                        "passage": {
                            "title": "East Ward Gazette",
                            "text": unrelated_span,
                        },
                    },
                },
            )
        )

    @staticmethod
    def _admitted_search_limit(
        adapter: QARetrievalReactExecutionAdapter,
        request: AgentRequest,
        observations: list[dict[str, object]],
    ) -> int:
        schema = adapter._state_conditioned_response_schema(
            request,
            observations,
        )
        assert schema is not None
        properties = schema["properties"]
        assert properties["name"]["const"] == "search"
        limit = properties["arguments"]["properties"]["limit"]["const"]
        assert type(limit) is int
        return limit

    def test_irrelevant_location_relation_read_preserves_typed_repair_and_retry(
        self,
    ) -> None:
        adapter, request, observations, _ = self._synthetic_location_repair_case(
            max_tool_calls=8,
        )
        initial_limit = self._admitted_search_limit(
            adapter,
            request,
            observations,
        )
        self._append_irrelevant_location_relation_search_and_read(
            observations,
            query="East Ward city Arcadia",
            limit=initial_limit,
        )

        state = adapter._required_evidence_state(request, observations)
        self.assertEqual("evidence", state.semantic_repair_kind)
        self.assertEqual("East Ward", state.location_containment_repair_anchor)
        self.assertEqual(1, state.location_containment_repair_search_count)
        self.assertEqual(0, state.location_containment_repair_read_count)
        self.assertEqual(
            (frozenset({(QA_RETRIEVAL_TOOL_ID, "search")}), False),
            adapter._state_conditioned_action_domain(request, observations),
        )

        retry_limit = self._admitted_search_limit(
            adapter,
            request,
            observations,
        )
        retry = StructuredAction(
            ActionKind.TOOL,
            "search",
            {"query": "East Ward part of Arcadia", "limit": retry_limit},
            resource_id=QA_RETRIEVAL_TOOL_ID,
        )
        self.assertIsNone(
            adapter._tool_action_error(
                request=request,
                action=retry,
                observations=observations,
            )
        )

        exhausted_adapter, exhausted_request, exhausted_observations, _ = (
            self._synthetic_location_repair_case(max_tool_calls=4)
        )
        exhausted_limit = self._admitted_search_limit(
            exhausted_adapter,
            exhausted_request,
            exhausted_observations,
        )
        self._append_irrelevant_location_relation_search_and_read(
            exhausted_observations,
            query="East Ward city Arcadia",
            limit=exhausted_limit,
        )
        self.assertEqual(
            (frozenset(), False),
            exhausted_adapter._state_conditioned_action_domain(
                exhausted_request,
                exhausted_observations,
            ),
        )

    async def test_location_relation_retry_remains_bounded_by_react_max_turns(
        self,
    ) -> None:
        _, base_request, _, first_hop_artifact = (
            self._synthetic_location_repair_case()
        )
        first_span = (
            "Professor Mira Hale was born in East Ward, North Province."
        )
        unrelated_span = (
            "The East Ward Gazette publishes a yearly arts calendar."
        )

        class FirstHopThenIrrelevantIndex(FakeIndex):
            def search(self, query: str, *, limit: int) -> tuple[FakeHit, ...]:
                self.search_calls.append((query, limit))
                if len(self.search_calls) == 1:
                    return (
                        FakeHit(
                            "first-hop",
                            "d1",
                            "Professor Mira Hale",
                            first_span,
                            1,
                        ),
                    )
                return (
                    FakeHit(
                        "unrelated-location-result",
                        "d2",
                        "East Ward Gazette",
                        unrelated_span,
                        1,
                    ),
                )

            def read(self, passage_id: str) -> FakePassage:
                self.read_calls.append(passage_id)
                if passage_id == "first-hop":
                    return FakePassage(
                        passage_id,
                        "d1",
                        "Professor Mira Hale",
                        first_span,
                    )
                return FakePassage(
                    passage_id,
                    "d2",
                    "East Ward Gazette",
                    unrelated_span,
                )

        def action(name: str, arguments: object) -> str:
            return json.dumps(
                {
                    "arguments": arguments,
                    "kind": "complete" if name == "complete" else "tool",
                    "name": name,
                    "resource_id": (
                        None if name == "complete" else QA_RETRIEVAL_TOOL_ID
                    ),
                    "skill_id": None,
                }
            )

        class SequenceGateway:
            def __init__(self) -> None:
                self.requests: list[AgentRequest] = []
                self.outputs = [
                    action(
                        "search",
                        {
                            "query": (
                                "Professor Mira Hale birth place Arcadia"
                            ),
                            "limit": 5,
                        },
                    ),
                    action("read", {"passage_id": "first-hop"}),
                    action("complete", {"value": first_hop_artifact}),
                    action(
                        "search",
                        {"query": "East Ward city Arcadia", "limit": 10},
                    ),
                    action(
                        "read",
                        {"passage_id": "unrelated-location-result"},
                    ),
                    action(
                        "search",
                        {"query": "East Ward part of Arcadia", "limit": 15},
                    ),
                ]

            async def generate(self, request: AgentRequest) -> AgentResponse:
                self.requests.append(request)
                return AgentResponse(self.outputs.pop(0))

        gateway = SequenceGateway()
        index = FirstHopThenIrrelevantIndex()
        with self.assertRaises(ReactExecutionError) as exhausted:
            await QARetrievalReactExecutionAdapter(
                gateway=gateway,
                tool_registry=build_qa_tool_registry(index),
                max_turns=5,
                max_tool_calls=8,
                task_type="factual_qa",
                completion_policy="required_evidence",
            ).execute(
                replace(
                    base_request,
                    request_id="trivia:location-retry-max-turns",
                )
            )

        self.assertIn("exhausted 5 turns", str(exhausted.exception))
        self.assertEqual(5, len(gateway.requests))
        self.assertEqual(5, len(exhausted.exception.react_trace))
        self.assertEqual(3, len(exhausted.exception.tool_receipts))
        self.assertEqual(
            "budget_exhausted",
            exhausted.exception.react_trace[-1]["observation_status"],
        )
        self.assertEqual(
            "retrieval_strategy_failure",
            exhausted.exception.react_trace[-1]["public_error_code"],
        )
        self.assertEqual(2, len(index.search_calls))
        self.assertEqual(
            ["first-hop"],
            index.read_calls,
        )

    def test_location_relation_query_requires_natural_lexical_relation(
        self,
    ) -> None:
        adapter, request, observations, _ = self._synthetic_location_repair_case()
        limit = self._admitted_search_limit(adapter, request, observations)

        rejected_queries = (
            (
                "Professor Mira Hale East Ward Arcadia "
                "administrative containment"
            ),
            "East Ward Arcadia administrative containment",
            "East Ward Arcadia geographic relation type",
        )
        for query in rejected_queries:
            with self.subTest(query=query):
                issue = adapter._tool_action_error(
                    request=request,
                    action=StructuredAction(
                        ActionKind.TOOL,
                        "search",
                        {"query": query, "limit": limit},
                        resource_id=QA_RETRIEVAL_TOOL_ID,
                    ),
                    observations=observations,
                )
                self.assertIsNotNone(issue)
                assert issue is not None
                self.assertIn(
                    "qa_location_relation_grounding_query_mismatch",
                    issue,
                )

        for query in (
            "East Ward city Arcadia",
            "East Ward part of Arcadia",
        ):
            with self.subTest(query=query):
                self.assertIsNone(
                    adapter._tool_action_error(
                        request=request,
                        action=StructuredAction(
                            ActionKind.TOOL,
                            "search",
                            {"query": query, "limit": limit},
                            resource_id=QA_RETRIEVAL_TOOL_ID,
                        ),
                        observations=observations,
                    )
                )

    def test_location_relation_retry_admits_rewrite_or_top_k_expansion(
        self,
    ) -> None:
        adapter, request, observations, _ = self._synthetic_location_repair_case(
            max_tool_calls=8,
        )
        first_limit = self._admitted_search_limit(
            adapter,
            request,
            observations,
        )
        first_query = "East Ward city Arcadia"
        self._append_irrelevant_location_relation_search_and_read(
            observations,
            query=first_query,
            limit=first_limit,
        )
        retry_limit = self._admitted_search_limit(
            adapter,
            request,
            observations,
        )
        self.assertGreater(retry_limit, first_limit)

        for query in (first_query, "East Ward part of Arcadia"):
            with self.subTest(query=query):
                self.assertIsNone(
                    adapter._tool_action_error(
                        request=request,
                        action=StructuredAction(
                            ActionKind.TOOL,
                            "search",
                            {"query": query, "limit": retry_limit},
                            resource_id=QA_RETRIEVAL_TOOL_ID,
                        ),
                        observations=observations,
                    )
                )

        off_schedule = adapter._tool_action_error(
            request=request,
            action=StructuredAction(
                ActionKind.TOOL,
                "search",
                {
                    "query": "East Ward part of Arcadia",
                    "limit": retry_limit + 1,
                },
                resource_id=QA_RETRIEVAL_TOOL_ID,
            ),
            observations=observations,
        )
        self.assertIsNotNone(off_schedule)
        assert off_schedule is not None
        self.assertIn("qa_retrieval_top_k_mismatch", off_schedule)

    def test_typed_location_continuation_precedes_factual_schedule_exhaustion(
        self,
    ) -> None:
        adapter, request, observations, _ = self._synthetic_location_repair_case(
            ordinary_strategy_count=5,
            max_tool_calls=14,
        )
        state = adapter._required_evidence_state(request, observations)
        self.assertEqual(5, state.strategy_progress_count)
        self.assertEqual("evidence", state.semantic_repair_kind)
        self.assertEqual("East Ward", state.location_containment_repair_anchor)
        self.assertEqual(
            (frozenset({(QA_RETRIEVAL_TOOL_ID, "search")}), False),
            adapter._state_conditioned_action_domain(request, observations),
        )

        limit = self._admitted_search_limit(adapter, request, observations)
        issue = adapter._tool_action_error(
            request=request,
            action=StructuredAction(
                ActionKind.TOOL,
                "search",
                {"query": "East Ward city Arcadia", "limit": limit},
                resource_id=QA_RETRIEVAL_TOOL_ID,
            ),
            observations=observations,
        )
        self.assertIsNone(issue)

    def test_exhausted_factual_schedule_uses_receipt_grounded_location_continuation(
        self,
    ) -> None:
        question = "Where in England was Dame Judi Dench born?"
        first_span = (
            "Dench was born in Heworth, North Riding of Yorkshire."
        )
        first_hop_artifact = {
            "question_scope": question,
            "answer_slot": {
                "answer_type": "location",
                "answer_cardinality": "single",
                "qualifiers": ["in England"],
                "proposition_index": 0,
                "answer_field": "object_or_attribute_value",
            },
            "evidence_propositions": [
                {
                    "subject": "Dench",
                    "relation": "was born in",
                    "object_or_attribute_value": (
                        "Heworth, North Riding of Yorkshire"
                    ),
                    "qualifiers": [],
                    "evidence_span": first_span,
                }
            ],
            "multi_hop_chain": [
                "Dench --was born in--> Heworth, North Riding of Yorkshire"
            ],
            "candidate_answer": "Heworth, North Riding of Yorkshire",
            "evidence": [first_span],
        }
        adapter = QARetrievalReactExecutionAdapter(
            gateway=SimpleNamespace(generate=lambda request: None),
            tool_registry=build_qa_tool_registry(FakeIndex()),
            max_turns=20,
            max_tool_calls=12,
            task_type="factual_qa",
            completion_policy="required_evidence",
        )
        request = AgentRequest(
            request_id="trivia:exhausted-location-repair",
            run_id="trivia",
            graph_revision=1,
            problem=question,
            agent=AgentNode(
                "reasoner",
                "model",
                "retrieve evidence and bind the requested location relation",
                role_family="reasoner",
                allowed_tools=(QA_RETRIEVAL_TOOL_ID,),
                execution_mode="react",
            ),
            model=ModelSpec("model", "provider"),
            provider=ProviderSpec("provider", kind="test"),
            phase=ExecutionPhase.SINGLE,
            semantic_protocol=QA_VERIFIED_ANSWER_LINEAGE_PROTOCOL,
        )

        queries = (
            "Dame Judi Dench birth place",
            "Dame Judi Dench birth place England",
            "Dame Judi Dench birthplace England",
            "Dame Judi Dench born place England",
            "Dame Judi Dench born England history",
        )
        observations: list[dict[str, object]] = []
        for index, (query, limit) in enumerate(
            zip(queries, (5, 10, 15, 20, 25)),
            start=1,
        ):
            passage_id = f"person-{index}"
            observations.extend(
                (
                    {
                        "observation_status": "success",
                        "executed_action": {
                            "kind": "tool",
                            "name": "search",
                            "resource_id": QA_RETRIEVAL_TOOL_ID,
                            "arguments": {"query": query, "limit": limit},
                        },
                        "result": {
                            "operation": "search",
                            "query": query,
                            "top_k": limit,
                            "passage_ids": [passage_id],
                            "hits": [
                                {
                                    "passage_id": passage_id,
                                    "title": "Judi Dench",
                                    "snippet": first_span,
                                    "rank": 1,
                                }
                            ],
                        },
                    },
                    {
                        "observation_status": "success",
                        "executed_action": {
                            "kind": "tool",
                            "name": "read",
                            "resource_id": QA_RETRIEVAL_TOOL_ID,
                            "arguments": {"passage_id": passage_id},
                        },
                        "result": {
                            "operation": "read",
                            "passage_id": passage_id,
                            "passage": {
                                "title": "Judi Dench",
                                "text": first_span,
                            },
                        },
                    },
                )
            )
        observations.append(
            {
                "observation_status": "schema_invalid",
                "public_error_code": (
                    "qa_semantic_evidence_provenance_invalid: "
                    "qa_location_containment_lineage_missing"
                ),
                "executed_action": {
                    "kind": "complete",
                    "name": "complete",
                    "resource_id": None,
                    "skill_id": None,
                    "arguments": {"value": first_hop_artifact},
                },
            }
        )

        state = adapter._required_evidence_state(request, observations)
        self.assertEqual(5, state.strategy_progress_count)
        self.assertEqual("Heworth", state.location_containment_repair_anchor)
        self.assertEqual(0, state.location_containment_repair_search_count)
        self.assertEqual(
            (frozenset({(QA_RETRIEVAL_TOOL_ID, "search")}), False),
            adapter._state_conditioned_action_domain(request, observations),
        )

        wrong_query = StructuredAction(
            ActionKind.TOOL,
            "search",
            {
                "query": "Dame Judi Dench born Heworth England",
                "limit": 25,
            },
            resource_id=QA_RETRIEVAL_TOOL_ID,
        )
        self.assertIn(
            "qa_location_relation_grounding_query_mismatch",
            adapter._tool_action_error(
                request=request,
                action=wrong_query,
                observations=observations,
            ),
        )
        grounding_query = StructuredAction(
            ActionKind.TOOL,
            "search",
            {"query": "Heworth city England", "limit": 25},
            resource_id=QA_RETRIEVAL_TOOL_ID,
        )
        self.assertIsNone(
            adapter._tool_action_error(
                request=request,
                action=grounding_query,
                observations=observations,
            )
        )
        response_schema = adapter._state_conditioned_response_schema(
            request,
            observations,
        )
        rendered_schema = json.dumps(response_schema, ensure_ascii=False)
        self.assertIn("Heworth", rendered_schema)
        self.assertIn("England", rendered_schema)
        self.assertNotIn("York", rendered_schema)
        self.assertNotIn("accepted_answers", rendered_schema)
        self.assertNotIn("ground_truth", rendered_schema)
        self.assertNotIn("evaluator", rendered_schema)

        containment_span = (
            "Heworth is part of the city of York in North Yorkshire, England."
        )
        observations.append(
            {
                "observation_status": "success",
                "executed_action": grounding_query.to_value(),
                "result": {
                    "operation": "search",
                    "query": "Heworth city England",
                    "top_k": 25,
                    "passage_ids": ["heworth-york"],
                    "hits": [
                        {
                            "passage_id": "heworth-york",
                            "title": "Heworth, York",
                            "snippet": containment_span,
                            "rank": 1,
                        }
                    ],
                },
            }
        )
        state_after_search = adapter._required_evidence_state(
            request,
            observations,
        )
        self.assertEqual(5, state_after_search.strategy_progress_count)
        self.assertEqual(
            1,
            state_after_search.location_containment_repair_search_count,
        )
        self.assertEqual(
            (frozenset({(QA_RETRIEVAL_TOOL_ID, "read")}), False),
            adapter._state_conditioned_action_domain(request, observations),
        )

        observations.append(
            {
                "observation_status": "success",
                "executed_action": {
                    "kind": "tool",
                    "name": "read",
                    "resource_id": QA_RETRIEVAL_TOOL_ID,
                    "arguments": {"passage_id": "heworth-york"},
                },
                "result": {
                    "operation": "read",
                    "passage_id": "heworth-york",
                    "passage": {
                        "title": "Heworth, York",
                        "text": containment_span,
                    },
                },
            }
        )
        final_state = adapter._required_evidence_state(request, observations)
        self.assertEqual(5, final_state.strategy_progress_count)
        self.assertIsNone(final_state.semantic_repair_kind)
        self.assertEqual(
            1,
            final_state.location_containment_repair_search_count,
        )
        self.assertEqual(
            1,
            final_state.location_containment_repair_read_count,
        )
        self.assertEqual(
            (frozenset(), True),
            adapter._state_conditioned_action_domain(request, observations),
        )
        completion_schema = adapter._state_conditioned_response_schema(
            request,
            observations,
        )
        assert completion_schema is not None
        completion_fields = completion_schema["properties"]["arguments"][
            "properties"
        ]["value"]["properties"]
        self.assertEqual(
            2,
            completion_fields["evidence_propositions"]["minItems"],
        )
        self.assertEqual(
            2,
            completion_fields["multi_hop_chain"]["minItems"],
        )
        self.assertIn(
            "resolution proposition",
            completion_fields["answer_slot"]["description"],
        )

        observations.append(
            {
                "observation_status": "schema_invalid",
                "public_error_code": (
                    "qa_semantic_evidence_provenance_invalid: "
                    "qa_location_containment_lineage_missing"
                ),
                "executed_action": {
                    "kind": "complete",
                    "name": "complete",
                    "resource_id": None,
                    "skill_id": None,
                    "arguments": {"value": first_hop_artifact},
                },
            }
        )
        repeated_state = adapter._required_evidence_state(
            request,
            observations,
        )
        self.assertEqual("structure", repeated_state.semantic_repair_kind)
        self.assertEqual(
            1,
            repeated_state.location_containment_repair_search_count,
        )
        self.assertEqual(
            1,
            repeated_state.location_containment_repair_read_count,
        )
        self.assertEqual(
            (frozenset(), True),
            adapter._state_conditioned_action_domain(request, observations),
        )
        continuation_state = adapter._public_retrieval_continuation_state(
            request,
            observations,
        )
        assert continuation_state is not None
        self.assertIsNone(continuation_state["required_strategy"])
        self.assertIsNone(continuation_state["required_top_k"])
        self.assertEqual(
            "structure",
            continuation_state["semantic_repair_kind"],
        )
        self.assertNotIn("required_operation", continuation_state)

    async def test_location_containment_repair_preserves_first_hop_and_is_bounded(
        self,
    ) -> None:
        question = "Where in Arcadia was Professor Mira Hale born?"
        first_span = (
            "Professor Mira Hale was born in East Ward, North Province."
        )
        containment_span = (
            "East Ward is a district and part of the city of Riverton in Arcadia."
        )

        class LocationIndex(FakeIndex):
            def search(self, query: str, *, limit: int) -> tuple[FakeHit, ...]:
                self.search_calls.append((query, limit))
                if len(self.search_calls) == 1:
                    return (
                        FakeHit("first-hop", "d1", "Mira Hale", first_span, 1),
                    )
                return (
                    FakeHit(
                        "containment-hop",
                        "d2",
                        "East Ward, Riverton",
                        containment_span,
                        1,
                    ),
                )

            def read(self, passage_id: str) -> FakePassage:
                self.read_calls.append(passage_id)
                if passage_id == "first-hop":
                    return FakePassage(
                        passage_id,
                        "d1",
                        "Mira Hale",
                        first_span,
                    )
                return FakePassage(
                    passage_id,
                    "d2",
                    "East Ward, Riverton",
                    containment_span,
                )

        first_hop = {
            "subject": "Professor Mira Hale",
            "relation": "was born in",
            "object_or_attribute_value": "East Ward, North Province",
            "qualifiers": [],
            "evidence_span": first_span,
        }
        one_hop_artifact = {
            "question_scope": question,
            "answer_slot": {
                "answer_type": "location",
                "answer_cardinality": "single",
                "qualifiers": ["in Arcadia"],
                "proposition_index": 0,
                "answer_field": "object_or_attribute_value",
            },
            "evidence_propositions": [first_hop],
            "multi_hop_chain": [
                "Professor Mira Hale --was born in--> East Ward, North Province"
            ],
            "candidate_answer": "East Ward, North Province",
            "evidence": [first_span],
        }
        valid_artifact = json.loads(json.dumps(one_hop_artifact))
        valid_artifact["answer_slot"]["proposition_index"] = 1
        valid_artifact["evidence_propositions"].append(
            {
                "subject": "East Ward",
                "relation": "is part of the city of",
                "object_or_attribute_value": "Riverton",
                "qualifiers": ["in Arcadia"],
                "evidence_span": containment_span,
            }
        )
        valid_artifact["multi_hop_chain"].append(
            "East Ward --part of the city of--> Riverton"
        )
        valid_artifact["candidate_answer"] = "Riverton"
        valid_artifact["evidence"].append(containment_span)

        def action(name: str, arguments: object) -> str:
            return json.dumps(
                {
                    "arguments": arguments,
                    "kind": "complete" if name == "complete" else "tool",
                    "name": name,
                    "resource_id": (
                        None if name == "complete" else QA_RETRIEVAL_TOOL_ID
                    ),
                    "skill_id": None,
                }
            )

        class SequenceGateway:
            def __init__(self) -> None:
                self.requests: list[AgentRequest] = []
                self.outputs = [
                    action(
                        "search",
                        {"query": "Professor Mira Hale birthplace", "limit": 5},
                    ),
                    action("read", {"passage_id": "first-hop"}),
                    action("complete", {"value": one_hop_artifact}),
                    action(
                        "search",
                        {"query": "East Ward city Arcadia", "limit": 10},
                    ),
                    action("read", {"passage_id": "containment-hop"}),
                    action("complete", {"value": valid_artifact}),
                ]

            async def generate(self, request: AgentRequest) -> AgentResponse:
                self.requests.append(request)
                return AgentResponse(self.outputs.pop(0))

        index = LocationIndex()
        gateway = SequenceGateway()
        request = AgentRequest(
            request_id="trivia:location-containment",
            run_id="trivia",
            graph_revision=1,
            problem=question,
            agent=AgentNode(
                "reasoner",
                "model",
                "retrieve evidence and bind the requested location relation",
                role_family="reasoner",
                allowed_tools=(QA_RETRIEVAL_TOOL_ID,),
                execution_mode="react",
            ),
            model=ModelSpec("model", "provider"),
            provider=ProviderSpec("provider", kind="test"),
            phase=ExecutionPhase.SINGLE,
            semantic_protocol=QA_VERIFIED_ANSWER_LINEAGE_PROTOCOL,
        )

        response = await QARetrievalReactExecutionAdapter(
            gateway=gateway,
            tool_registry=build_qa_tool_registry(index),
            max_turns=6,
            max_tool_calls=4,
            task_type="factual_qa",
            completion_policy="required_evidence",
        ).execute(request)

        self.assertEqual(
            json.dumps(
                valid_artifact,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ),
            response.text,
        )
        self.assertEqual(
            [
                ("Professor Mira Hale birthplace", 5),
                ("East Ward city Arcadia", 10),
            ],
            index.search_calls,
        )
        self.assertEqual(["first-hop", "containment-hop"], index.read_calls)
        self.assertEqual(4, response.metadata["tool_calls"])
        feedback = response.metadata["react_trace"][2]["public_error_code"]
        self.assertEqual(
            "qa_semantic_evidence_provenance_invalid: "
            "qa_location_containment_lineage_missing",
            feedback,
        )
        repair_contract = gateway.requests[3].agent.contract
        self.assertIn("Preserve the receipt-grounded first-hop", repair_contract)
        self.assertIn("Currently admissible Tool action contracts", repair_contract)
        self.assertNotIn("accepted_answers", repair_contract)
        self.assertNotIn("ground_truth", repair_contract)

        class RepeatedStructuredFailureGateway:
            def __init__(self) -> None:
                self.outputs = [
                    action(
                        "search",
                        {"query": "Professor Mira Hale birthplace", "limit": 5},
                    ),
                    action("read", {"passage_id": "first-hop"}),
                    action("complete", {"value": one_hop_artifact}),
                    action(
                        "search",
                        {"query": "East Ward city Arcadia", "limit": 10},
                    ),
                    action("read", {"passage_id": "containment-hop"}),
                    action("complete", {"value": one_hop_artifact}),
                ]

            async def generate(self, request: AgentRequest) -> AgentResponse:
                del request
                return AgentResponse(self.outputs.pop(0))

        structured_adapter = QARetrievalReactExecutionAdapter(
                gateway=RepeatedStructuredFailureGateway(),
                tool_registry=build_qa_tool_registry(LocationIndex()),
                max_turns=6,
                max_tool_calls=4,
                task_type="factual_qa",
                completion_policy="required_evidence",
            )
        with self.assertRaises(ReactExecutionError) as structured_exhaustion:
            await structured_adapter.execute(
                replace(
                    request,
                    request_id="trivia:containment-structured-repair",
                )
            )
        structured_state = structured_adapter._required_evidence_state(
            request,
            structured_adapter._continuation_observations(
                structured_exhaustion.exception.react_trace
            ),
        )
        self.assertEqual("structure", structured_state.semantic_repair_kind)
        last_structured_observation = (
            structured_exhaustion.exception.react_trace[-1]
        )
        self.assertEqual(
            "qa_semantic_artifact_invalid: "
            "qa_location_containment_lineage_missing",
            last_structured_observation["public_error_code"],
        )
        self.assertIn(
            "Preserve the receipt-grounded first-hop",
            last_structured_observation["repair_instruction"],
        )
        self.assertNotIn(
            "terminal_failure_diagnosis",
            last_structured_observation,
        )
        self.assertFalse(structured_exhaustion.exception.tool_plan_exhausted)

        country_span = "East Ward belongs to the country of Arcadia."
        invalid_promotion = json.loads(json.dumps(one_hop_artifact))
        invalid_promotion["answer_slot"]["proposition_index"] = 1
        invalid_promotion["evidence_propositions"].append(
            {
                "subject": "East Ward",
                "relation": "belongs to the country of",
                "object_or_attribute_value": "Arcadia",
                "qualifiers": [],
                "evidence_span": country_span,
            }
        )
        invalid_promotion["multi_hop_chain"].append(
            "East Ward --belongs to the country of--> Arcadia"
        )
        invalid_promotion["candidate_answer"] = "Arcadia"
        invalid_promotion["evidence"].append(country_span)

        class CountryPromotionIndex(LocationIndex):
            def search(self, query: str, *, limit: int) -> tuple[FakeHit, ...]:
                self.search_calls.append((query, limit))
                if len(self.search_calls) == 1:
                    return (
                        FakeHit("first-hop", "d1", "Mira Hale", first_span, 1),
                    )
                return (
                    FakeHit("country-hop", "d3", "East Ward", country_span, 1),
                )

            def read(self, passage_id: str) -> FakePassage:
                self.read_calls.append(passage_id)
                if passage_id == "first-hop":
                    return FakePassage(
                        passage_id,
                        "d1",
                        "Mira Hale",
                        first_span,
                    )
                return FakePassage(
                    passage_id,
                    "d3",
                    "East Ward",
                    country_span,
                )

        class NegativeGateway:
            def __init__(self) -> None:
                self.outputs = [
                    action(
                        "search",
                        {"query": "Professor Mira Hale birthplace", "limit": 5},
                    ),
                    action("read", {"passage_id": "first-hop"}),
                    action("complete", {"value": one_hop_artifact}),
                    action(
                        "search",
                        {
                            "query": "East Ward part country Arcadia",
                            "limit": 10,
                        },
                    ),
                    action("read", {"passage_id": "country-hop"}),
                    action("complete", {"value": invalid_promotion}),
                ]

            async def generate(self, request: AgentRequest) -> AgentResponse:
                del request
                return AgentResponse(self.outputs.pop(0))

        negative_index = CountryPromotionIndex()
        with self.assertRaises(ReactExecutionError) as exhausted:
            await QARetrievalReactExecutionAdapter(
                gateway=NegativeGateway(),
                tool_registry=build_qa_tool_registry(negative_index),
                max_turns=6,
                max_tool_calls=4,
                task_type="factual_qa",
                completion_policy="required_evidence",
            ).execute(replace(request, request_id="trivia:country-not-city"))
        self.assertEqual(2, len(negative_index.search_calls))
        self.assertEqual(["first-hop", "country-hop"], negative_index.read_calls)
        terminal_error = exhausted.exception.react_trace[-1]["public_error_code"]
        self.assertEqual("retrieval_strategy_failure", terminal_error)
        self.assertNotIn("Arcadia", terminal_error)

    def test_factual_query_guidance_is_scope_preserving_and_sample_neutral(
        self,
    ) -> None:
        adapter = QARetrievalReactExecutionAdapter(
            gateway=SimpleNamespace(generate=lambda request: None),
            tool_registry=build_qa_tool_registry(FakeIndex()),
            max_turns=16,
            max_tool_calls=12,
            task_type="factual_qa",
            completion_policy="required_evidence",
        )
        request = AgentRequest(
            request_id="factual:neutral-query-guidance",
            run_id="factual",
            graph_revision=1,
            problem="Which institution established the annual index?",
            agent=AgentNode(
                "retriever",
                "model",
                "retrieve public evidence for the original entity and relation",
                role_family="evidence_retriever",
                allowed_tools=(QA_RETRIEVAL_TOOL_ID,),
                execution_mode="react",
            ),
            model=ModelSpec("model", "provider"),
            provider=ProviderSpec("provider", kind="test"),
            phase=ExecutionPhase.SINGLE,
            semantic_protocol=QA_VERIFIED_ANSWER_LINEAGE_PROTOCOL,
        )
        expected_guidance = (
            (
                "initial_retrieval",
                (
                    "target entity/topic anchor",
                    "answer-slot wording",
                    "auxiliary verbs",
                ),
            ),
            (
                "spelling_normalization",
                (
                    "Correct spelling, tokenization, and grammatical noise",
                    "do not append a strategy label",
                    "introduce a new topic",
                ),
            ),
            (
                "alias_expansion",
                (
                    "discovery query rather than evidence",
                    "established domain expression",
                    "returned title/snippet",
                    "same entity/topic",
                    "unrelated subtype",
                ),
            ),
            (
                "entity_disambiguation",
                (
                    "exact title anchor",
                    "title and its snippet jointly identify",
                    "irrelevant subtopic",
                ),
            ),
            (
                "query_rewriting",
                (
                    "exact chosen entity anchor",
                    "verbal and nominal paraphrases",
                    "same predicate meaning",
                ),
            ),
        )
        queries = (
            "institution establish annual index",
            "institution established annual index",
            "organization founding yearly index",
            "publisher annual index origin",
        )
        observations: list[dict[str, object]] = []
        rendered_guidance: list[str] = []
        for strategy_index, (strategy, fragments) in enumerate(
            expected_guidance
        ):
            schema = adapter._state_conditioned_response_schema(
                request,
                observations,
            )
            assert schema is not None
            query_description = (
                schema["properties"]["arguments"]["properties"]
                ["query"]["description"]
            )
            contract = adapter._contract(request, observations)
            rendered_guidance.extend((query_description, contract))
            self.assertIn(strategy, query_description)
            self.assertIn(f"`{strategy}`", contract)
            self.assertIn("content-bearing entity/relation terms", query_description)
            self.assertIn("unrelated topic", query_description)
            for fragment in fragments:
                self.assertIn(fragment, query_description)
                self.assertIn(fragment, contract)

            if strategy_index == len(queries):
                continue
            passage_id = f"p{strategy_index}"
            limit = (5, 10, 15, 20)[strategy_index]
            observations.extend(
                (
                    {
                        "observation_status": "success",
                        "executed_action": {
                            "kind": "tool",
                            "name": "search",
                            "resource_id": QA_RETRIEVAL_TOOL_ID,
                            "arguments": {
                                "query": queries[strategy_index],
                                "limit": limit,
                            },
                        },
                        "result": {
                            "operation": "search",
                            "query": queries[strategy_index],
                            "top_k": limit,
                            "passage_ids": [passage_id],
                        },
                    },
                    {
                        "observation_status": "success",
                        "executed_action": {
                            "kind": "tool",
                            "name": "read",
                            "resource_id": QA_RETRIEVAL_TOOL_ID,
                            "arguments": {"passage_id": passage_id},
                        },
                        "result": {
                            "operation": "read",
                            "passage_id": passage_id,
                            "passage": {
                                "title": "Public index",
                                "text": "Public but relation-insufficient evidence.",
                            },
                        },
                    },
                    {
                        "observation_status": "schema_invalid",
                        "public_error_code": (
                            "qa_semantic_evidence_provenance_invalid: "
                            "target relation absent"
                        ),
                    },
                )
            )

        combined = "\n".join(rendered_guidance)
        for sample_recipe in (
            "Billboard",
            "1936",
            "Easy Listening",
            "Hot 100",
            "TriviaQA",
            "HotpotQA",
        ):
            self.assertNotIn(sample_recipe, combined)

        search_observation = {
            "observation_status": "success",
            "executed_action": {
                "kind": "tool",
                "name": "search",
                "resource_id": QA_RETRIEVAL_TOOL_ID,
                "arguments": {
                    "query": "institution establish annual index",
                    "limit": 5,
                },
            },
            "result": {
                "operation": "search",
                "query": "institution establish annual index",
                "top_k": 5,
                "passage_ids": ["p0", "p1"],
            },
        }
        read_contract = adapter._contract(request, [search_observation])
        self.assertIn("title and snippet jointly", read_contract)
        self.assertIn("Rank is retrieval order, not evidence or proof", read_contract)

    def test_factual_strategy_requires_new_fts_terms_and_preserves_ordinal_scope(
        self,
    ) -> None:
        adapter = QARetrievalReactExecutionAdapter(
            gateway=SimpleNamespace(generate=lambda request: None),
            tool_registry=build_qa_tool_registry(FakeIndex()),
            max_turns=16,
            max_tool_calls=12,
            task_type="factual_qa",
            completion_policy="required_evidence",
        )
        request = AgentRequest(
            request_id="trivia:query-semantic-delta",
            run_id="trivia",
            graph_revision=1,
            problem=(
                "In which decade did Billboard magazine first publish an "
                "American hit chart?"
            ),
            agent=AgentNode(
                "retriever",
                "model",
                "retrieve question-grounded evidence",
                role_family="evidence_retriever",
                allowed_tools=(QA_RETRIEVAL_TOOL_ID,),
                execution_mode="react",
            ),
            model=ModelSpec("model", "provider"),
            provider=ProviderSpec("provider", kind="test"),
            phase=ExecutionPhase.SINGLE,
            semantic_protocol=QA_VERIFIED_ANSWER_LINEAGE_PROTOCOL,
        )
        observations = [
            {
                "observation_status": "success",
                "executed_action": {
                    "kind": "tool",
                    "name": "search",
                    "resource_id": QA_RETRIEVAL_TOOL_ID,
                    "arguments": {
                        "query": (
                            "Billboard magazine first publish American hit chart"
                        ),
                        "limit": 5,
                    },
                },
                "result": {
                    "operation": "search",
                    "query": (
                        "Billboard magazine first publish American hit chart"
                    ),
                    "top_k": 5,
                    "passage_ids": ["p1"],
                },
            }
        ]

        reordered = StructuredAction(
            ActionKind.TOOL,
            "search",
            {
                "query": (
                    "Billboard magazine American hit chart first publish"
                ),
                "limit": 10,
            },
            resource_id=QA_RETRIEVAL_TOOL_ID,
        )
        self.assertEqual(
            "qa_retrieval_duplicate_normalized_query",
            adapter._tool_action_error(
                request=request,
                action=reordered,
                observations=observations,
            ),
        )
        scope_dropped = StructuredAction(
            ActionKind.TOOL,
            "search",
            {
                "query": (
                    "Billboard magazine publish national music hit parade origin"
                ),
                "limit": 10,
            },
            resource_id=QA_RETRIEVAL_TOOL_ID,
        )
        self.assertEqual(
            "qa_retrieval_query_scope_modifier_loss",
            adapter._tool_action_error(
                request=request,
                action=scope_dropped,
                observations=observations,
            ),
        )
        scope_alias = StructuredAction(
            ActionKind.TOOL,
            "search",
            {
                "query": (
                    "Billboard magazine inaugural publish American hit chart"
                ),
                "limit": 10,
            },
            resource_id=QA_RETRIEVAL_TOOL_ID,
        )
        self.assertIsNone(
            adapter._tool_action_error(
                request=request,
                action=scope_alias,
                observations=observations,
            )
        )
        relation_alias = StructuredAction(
            ActionKind.TOOL,
            "search",
            {
                "query": (
                    "Billboard magazine first publish music hit parade"
                ),
                "limit": 10,
            },
            resource_id=QA_RETRIEVAL_TOOL_ID,
        )
        self.assertIsNone(
            adapter._tool_action_error(
                request=request,
                action=relation_alias,
                observations=observations,
            )
        )
        self.assertIn(
            "Restore every explicit ordinal scope class",
            adapter._public_semantic_repair_instruction(
                "qa_retrieval_query_scope_modifier_loss"
            ),
        )

    def test_public_hit_metadata_conditions_read_or_search_without_changing_wire(
        self,
    ) -> None:
        request = AgentRequest(
            request_id="trivia:public-search-candidates",
            run_id="trivia",
            graph_revision=1,
            problem="Which institution first published the annual index?",
            agent=AgentNode(
                "retriever",
                "model",
                "retrieve question-grounded evidence",
                role_family="evidence_retriever",
                allowed_tools=(QA_RETRIEVAL_TOOL_ID,),
                execution_mode="react",
            ),
            model=ModelSpec("model", "provider"),
            provider=ProviderSpec("provider", kind="test"),
            phase=ExecutionPhase.SINGLE,
            semantic_protocol=QA_VERIFIED_ANSWER_LINEAGE_PROTOCOL,
        )
        observations = [
            {
                "observation_status": "success",
                "executed_action": {
                    "kind": "tool",
                    "name": "search",
                    "resource_id": QA_RETRIEVAL_TOOL_ID,
                    "arguments": {
                        "query": "institution first annual index publish",
                        "limit": 5,
                    },
                },
                "result": {
                    "operation": "search",
                    "query": "institution first annual index publish",
                    "top_k": 5,
                    "passage_ids": ["p1", "p2"],
                    "hits": [
                        {
                            "passage_id": "p1",
                            "document_id": "d1",
                            "title": "Unrelated annual report",
                            "snippet": "A different institution issued a report.",
                            "rank": 1,
                        },
                        {
                            "passage_id": "p2",
                            "document_id": "d2",
                            "title": "Target institution",
                            "snippet": "The institution first published the annual index.",
                            "rank": 2,
                        },
                    ],
                },
            }
        ]
        adapter = QARetrievalReactExecutionAdapter(
            gateway=SimpleNamespace(generate=lambda request: None),
            tool_registry=build_qa_tool_registry(FakeIndex()),
            max_turns=16,
            max_tool_calls=12,
            task_type="factual_qa",
            completion_policy="required_evidence",
        )
        actions, completion = adapter._state_conditioned_action_domain(
            request,
            observations,
        )
        self.assertEqual(
            frozenset(
                {
                    (QA_RETRIEVAL_TOOL_ID, "read"),
                    (QA_RETRIEVAL_TOOL_ID, "search"),
                }
            ),
            actions,
        )
        self.assertFalse(completion)
        contract = adapter._contract(request, observations)
        # The lossless public search Observation remains in the prompt, while
        # the current read action domain and compact candidate projection are
        # narrowed to the compatible rank-2 hit.
        self.assertIn("Unrelated annual report", contract)
        self.assertIn("Target institution", contract)
        self.assertIn(
            'For read, arguments contains only one passage_id from: ["p2"]',
            contract,
        )
        self.assertIn("when none does, issue a new qa-retrieval search", contract)

        bounded_adapter = QARetrievalReactExecutionAdapter(
            gateway=SimpleNamespace(generate=lambda request: None),
            tool_registry=build_qa_tool_registry(FakeIndex()),
            max_turns=4,
            max_tool_calls=2,
            task_type="factual_qa",
            completion_policy="required_evidence",
        )
        read_schema = bounded_adapter._state_conditioned_response_schema(
            request,
            observations,
        )
        assert read_schema is not None
        passage_schema = read_schema["properties"]["arguments"]["properties"][
            "passage_id"
        ]
        self.assertEqual(["p2"], passage_schema["enum"])
        self.assertIn("Target institution", passage_schema["description"])
        self.assertIn(
            "Rank is retrieval order, not evidence or proof",
            passage_schema["description"],
        )

        completion_schema = adapter._completion_arguments_schema(request)
        relation_description = completion_schema["properties"]["value"][
            "properties"
        ]["target_relation"]["description"]
        self.assertIn("question-side requested relation", relation_description)
        self.assertIn("question-side", relation_description)
        self.assertIn("receipt-grounded", relation_description)
        self.assertIn(
            "replace it with the wording from the read receipt",
            relation_description,
        )
        serialized_completion_schema = json.dumps(completion_schema)
        for answer_field in (
            "accepted_answers",
            "candidate_answer",
            "final_answer",
            "gold_answer",
        ):
            self.assertNotIn(answer_field, serialized_completion_schema)

    def test_v23_composite_relation_rejects_bare_head_and_trace_drift(
        self,
    ) -> None:
        question = (
            "In which decade did Chart Weekly magazine first publish an "
            "American hit chart?"
        )
        initial_query = (
            "Chart Weekly magazine first publish American hit chart decade"
        )
        self.assertEqual(
            ("chart", "weekly", "magazine"),
            _question_entity_anchor_tokens(question),
        )
        self.assertFalse(
            _query_replaces_relation_surface(
                original_question=question,
                previous_query=initial_query,
                query=(
                    "Chart Weekly magazine first publish American chart decade"
                ),
            )
        )
        self.assertTrue(
            _query_replaces_relation_surface(
                original_question=question,
                previous_query=initial_query,
                query=(
                    "Chart Weekly magazine first publish American music hit "
                    "parade decade"
                ),
            )
        )

        measured_trace = (
            initial_query,
            "Chart Weekly magazine first American hit chart decade",
            (
                "Chart Weekly magazine first weekly American popular music "
                "chart history decade"
            ),
            (
                "Chart Weekly magazine first weekly American popular music "
                "history chart"
            ),
            "Chart Weekly magazine first American hit chart history",
        )
        self.assertEqual(
            (True, False, False, False, False),
            _factual_strategy_semantics_verified(
                original_question=question,
                distinct_queries=measured_trace,
            ),
        )

        relation_coverage_trace = (
            initial_query,
            (
                "Chart Weekly magazine first published American hit chart "
                "decade"
            ),
            (
                "Chart Weekly magazine first publication American hit chart "
                "decade"
            ),
            (
                "Chart Weekly magazine first publication American hit chart "
                "history decade"
            ),
            (
                "Chart Weekly magazine first publication American music hit "
                "parade history decade"
            ),
        )
        self.assertEqual(
            (True, True, True, True, True),
            _factual_strategy_semantics_verified(
                original_question=question,
                distinct_queries=relation_coverage_trace,
            ),
        )
        repeated_class_trace = (
            *relation_coverage_trace[:-1],
            (
                "Chart Weekly magazine first published American hit chart "
                "history decade"
            ),
        )
        self.assertEqual(
            (True, True, True, True, False),
            _factual_strategy_semantics_verified(
                original_question=question,
                distinct_queries=repeated_class_trace,
            ),
        )

    def test_v23_public_action_domain_exposes_answer_free_relation_surfaces(
        self,
    ) -> None:
        question = (
            "In which decade did Chart Weekly magazine first publish an "
            "American hit chart?"
        )
        request = AgentRequest(
            request_id="factual:v23-public-relation-domain",
            run_id="factual",
            graph_revision=1,
            problem=question,
            agent=AgentNode(
                "retriever",
                "model",
                "retrieve public evidence for the requested relation",
                role_family="evidence_retriever",
                allowed_tools=(QA_RETRIEVAL_TOOL_ID,),
                execution_mode="react",
            ),
            model=ModelSpec("model", "provider"),
            provider=ProviderSpec("provider", kind="test"),
            phase=ExecutionPhase.SINGLE,
            semantic_protocol=QA_VERIFIED_ANSWER_LINEAGE_PROTOCOL,
        )
        adapter = QARetrievalReactExecutionAdapter(
            gateway=SimpleNamespace(generate=lambda request: None),
            tool_registry=build_qa_tool_registry(FakeIndex()),
            max_turns=16,
            max_tool_calls=12,
            task_type="factual_qa",
            completion_policy="required_evidence",
        )
        public_state = adapter._public_retrieval_continuation_state(request, [])
        assert public_state is not None
        self.assertEqual(
            ["hit_chart", "publication"],
            public_state["required_relation_classes"],
        )
        self.assertEqual(
            ["hit chart", "hit parade", "music hit parade"],
            public_state["required_relation_surface_alternatives"]["hit_chart"],
        )
        self.assertNotIn(
            "chart",
            public_state["required_relation_surface_alternatives"]["hit_chart"],
        )
        schema = adapter._state_conditioned_response_schema(request, [])
        assert schema is not None
        query_schema = schema["properties"]["arguments"]["properties"]["query"]
        self.assertNotIn("enum", query_schema)
        self.assertIn("music hit parade", query_schema["description"])

        missing_relation = StructuredAction(
            ActionKind.TOOL,
            "search",
            {
                "query": "Chart Weekly magazine first American hit chart",
                "limit": 5,
            },
            resource_id=QA_RETRIEVAL_TOOL_ID,
        )
        issue = adapter._tool_action_error(
            request=request,
            action=missing_relation,
            observations=[],
        )
        self.assertIsNotNone(issue)
        assert issue is not None
        self.assertTrue(issue.startswith("qa_retrieval_query_target_relation_loss"))
        self.assertIn("publication", issue)
        self.assertIn(
            "required_relation_surface_alternatives",
            adapter._public_semantic_repair_instruction(issue),
        )

        prior_queries = (
            (
                "Chart Weekly magazine first publish American hit chart decade",
                5,
            ),
            (
                "Chart Weekly magazine first published American hit chart decade",
                10,
            ),
            (
                "Chart Weekly magazine first publication American hit chart decade",
                15,
            ),
            (
                "Chart Weekly magazine first publication American hit chart "
                "history decade",
                20,
            ),
        )
        observations = [
            {
                "observation_status": "success",
                "executed_action": {
                    "kind": "tool",
                    "name": "search",
                    "resource_id": QA_RETRIEVAL_TOOL_ID,
                    "arguments": {"query": query, "limit": limit},
                },
                "result": {
                    "operation": "search",
                    "query": query,
                    "top_k": limit,
                    "passage_ids": [],
                },
            }
            for query, limit in prior_queries
        ]
        coverage_state = adapter._public_retrieval_continuation_state(
            request,
            observations,
        )
        assert coverage_state is not None
        self.assertEqual(
            ["publication"],
            coverage_state["transformed_relation_classes"],
        )
        self.assertEqual(
            ["hit_chart"],
            coverage_state["remaining_relation_transformation_classes"],
        )

        repeated_publication = StructuredAction(
            ActionKind.TOOL,
            "search",
            {
                "query": (
                    "Chart Weekly magazine first published American hit chart "
                    "history decade"
                ),
                "limit": 25,
            },
            resource_id=QA_RETRIEVAL_TOOL_ID,
        )
        repeated_issue = adapter._tool_action_error(
            request=request,
            action=repeated_publication,
            observations=observations,
        )
        self.assertIsNotNone(repeated_issue)
        assert repeated_issue is not None
        self.assertTrue(
            repeated_issue.startswith(
                "qa_retrieval_query_strategy_semantics_mismatch"
            )
        )
        self.assertIn("hit_chart", repeated_issue)
        self.assertIn(
            "remaining_relation_transformation_classes",
            adapter._public_semantic_repair_instruction(repeated_issue),
        )

        new_hit_chart_surface = StructuredAction(
            ActionKind.TOOL,
            "search",
            {
                "query": (
                    "Chart Weekly magazine first publication American music hit "
                    "parade history decade"
                ),
                "limit": 25,
            },
            resource_id=QA_RETRIEVAL_TOOL_ID,
        )
        self.assertIsNone(
            adapter._tool_action_error(
                request=request,
                action=new_hit_chart_surface,
                observations=observations,
            )
        )

    def test_v23_strong_candidate_binds_entity_composite_relation_and_ordinal(
        self,
    ) -> None:
        question = (
            "In which decade did Chart Weekly magazine first publish an "
            "American hit chart?"
        )
        supported = _public_search_candidate_compatibility(
            original_question=question,
            title="Hit parade",
            snippet=(
                "Chart Weekly magazine published its first music hit parade "
                "on a winter date."
            ),
        )
        subtype = _public_search_candidate_compatibility(
            original_question=question,
            title="Chart Weekly Live",
            snippet=(
                "The Chart Weekly music magazine charts were first published "
                "for an interactive service."
            ),
        )
        regional = _public_search_candidate_compatibility(
            original_question=question,
            title="Regional Airplay",
            snippet=(
                "Regional Airplay is a record chart published weekly by Chart "
                "Weekly magazine. According to its database, the first chart "
                "was published for one region."
            ),
        )
        self.assertTrue(supported[0])
        self.assertFalse(subtype[0])
        self.assertFalse(regional[0])

    def test_ordinal_read_domain_keeps_entity_title_when_snippet_is_truncated(
        self,
    ) -> None:
        question = (
            "In which decade did Chart Weekly magazine first publish an "
            "American hit chart?"
        )
        observations = [
            {
                "observation_status": "success",
                "result": {
                    "operation": "search",
                    "hits": [
                        {
                            "passage_id": "entity-title",
                            "rank": 2,
                            "title": "Chart Weekly (magazine)",
                            "snippet": (
                                "The publication expanded into radio coverage "
                                "during the interwar period…"
                            ),
                        },
                        {
                            "passage_id": "distractor",
                            "rank": 3,
                            "title": "A regional performer",
                            "snippet": (
                                "Chart Weekly magazine later listed the act."
                            ),
                        },
                    ],
                },
            }
        ]

        candidates = (
            QARetrievalReactExecutionAdapter._latest_public_search_candidates(
                observations,
                unread_passage_ids=("entity-title", "distractor"),
                original_question=question,
            )
        )

        self.assertEqual(
            ("entity-title",),
            tuple(candidate["passage_id"] for candidate in candidates),
        )

    def test_v21_relation_strategy_replaces_instead_of_appending_alias(self) -> None:
        question = (
            "In which decade did Chart Weekly magazine first publish an "
            "American hit chart?"
        )
        request = AgentRequest(
            request_id="factual:v21-relation-replacement",
            run_id="factual",
            graph_revision=1,
            problem=question,
            agent=AgentNode(
                "retriever",
                "model",
                "retrieve public evidence for the requested relation",
                role_family="evidence_retriever",
                allowed_tools=(QA_RETRIEVAL_TOOL_ID,),
                execution_mode="react",
            ),
            model=ModelSpec("model", "provider"),
            provider=ProviderSpec("provider", kind="test"),
            phase=ExecutionPhase.SINGLE,
            semantic_protocol=QA_VERIFIED_ANSWER_LINEAGE_PROTOCOL,
        )
        adapter = QARetrievalReactExecutionAdapter(
            gateway=SimpleNamespace(generate=lambda request: None),
            tool_registry=build_qa_tool_registry(FakeIndex()),
            max_turns=16,
            max_tool_calls=12,
            task_type="factual_qa",
            completion_policy="required_evidence",
        )

        def search_observation(query: str, limit: int) -> dict[str, object]:
            return {
                "observation_status": "success",
                "executed_action": {
                    "kind": "tool",
                    "name": "search",
                    "resource_id": QA_RETRIEVAL_TOOL_ID,
                    "arguments": {"query": query, "limit": limit},
                },
                "result": {
                    "operation": "search",
                    "query": query,
                    "top_k": limit,
                    "passage_ids": [],
                    "hits": [],
                },
            }

        observations = [
            search_observation(
                "Chart Weekly magazine first publish American hit chart",
                5,
            ),
            search_observation(
                "Chart Weekly magazine first published American hit chart",
                10,
            ),
        ]
        appended = StructuredAction(
            ActionKind.TOOL,
            "search",
            {
                "query": (
                    "Chart Weekly magazine first published American hit chart "
                    "music hit parade"
                ),
                "limit": 15,
            },
            resource_id=QA_RETRIEVAL_TOOL_ID,
        )
        appended_issue = adapter._tool_action_error(
            request=request,
            action=appended,
            observations=observations,
        )
        self.assertIsNotNone(appended_issue)
        assert appended_issue is not None
        self.assertTrue(
            appended_issue.startswith(
                "qa_retrieval_query_strategy_semantics_mismatch"
            )
        )
        self.assertIn("remaining_relation_classes", appended_issue)
        alias_replacement = StructuredAction(
            ActionKind.TOOL,
            "search",
            {
                "query": (
                    "Chart Weekly magazine first published American music hit parade"
                ),
                "limit": 15,
            },
            resource_id=QA_RETRIEVAL_TOOL_ID,
        )
        self.assertIsNone(
            adapter._tool_action_error(
                request=request,
                action=alias_replacement,
                observations=observations,
            )
        )

        observations.extend(
            (
                search_observation(
                    (
                        "Chart Weekly magazine first published American "
                        "music hit parade"
                    ),
                    15,
                ),
                search_observation(
                    (
                        "Chart Weekly magazine publication first American "
                        "music hit parade"
                    ),
                    20,
                ),
            )
        )
        rewrite_replacement = StructuredAction(
            ActionKind.TOOL,
            "search",
            {
                "query": (
                    "Chart Weekly magazine published first American music hit parade history"
                ),
                "limit": 25,
            },
            resource_id=QA_RETRIEVAL_TOOL_ID,
        )
        self.assertIsNone(
            adapter._tool_action_error(
                request=request,
                action=rewrite_replacement,
                observations=observations,
            )
        )
        state = adapter._required_evidence_state(
            request,
            [
                *observations,
                search_observation(
                    (
                        "Chart Weekly magazine published first American "
                        "music hit parade history"
                    ),
                    25,
                ),
            ],
        )
        self.assertEqual((True, True, True, True, True), state.strategy_semantics)
        self.assertTrue(state.strategy_semantics_verified)
        self.assertEqual(
            "knowledge_base_coverage_failure",
            adapter._factual_exhaustion_diagnosis(
                strategy_progress_count=5,
                strategy_semantics_verified=True,
                successful_search_hit_counts=(0, 0, 0, 0, 0),
                tool_error_count=0,
            ),
        )
        self.assertEqual(
            "retrieval_recall_failure",
            adapter._factual_exhaustion_diagnosis(
                strategy_progress_count=5,
                strategy_semantics_verified=True,
                successful_search_hit_counts=(0, 0, 1, 0, 0),
                tool_error_count=0,
            ),
        )

    def test_v21_joint_schema_filters_local_first_hit_false_positives(self) -> None:
        question = (
            "In which decade did Chart Weekly magazine first publish an "
            "American hit chart?"
        )
        request = AgentRequest(
            request_id="factual:v21-public-candidate-domain",
            run_id="factual",
            graph_revision=1,
            problem=question,
            agent=AgentNode(
                "retriever",
                "model",
                "retrieve question-grounded public evidence",
                role_family="evidence_retriever",
                allowed_tools=(QA_RETRIEVAL_TOOL_ID,),
                execution_mode="react",
            ),
            model=ModelSpec("model", "provider"),
            provider=ProviderSpec("provider", kind="test"),
            phase=ExecutionPhase.SINGLE,
            semantic_protocol=QA_VERIFIED_ANSWER_LINEAGE_PROTOCOL,
        )
        observations = [
            {
                "observation_status": "success",
                "executed_action": {
                    "kind": "tool",
                    "name": "search",
                    "resource_id": QA_RETRIEVAL_TOOL_ID,
                    "arguments": {
                        "query": "Chart Weekly first American hit chart",
                        "limit": 5,
                    },
                },
                "result": {
                    "operation": "search",
                    "query": "Chart Weekly first American hit chart",
                    "top_k": 5,
                    "passage_ids": ["p-hit", "p-sarah", "p-weekly", "p-patti"],
                    "hits": [
                        {
                            "passage_id": "p-hit",
                            "document_id": "d1",
                            "title": "Hit parade",
                            "snippet": (
                                "Chart Weekly magazine first published an "
                                "American music hit parade in its listings."
                            ),
                            "rank": 1,
                        },
                        {
                            "passage_id": "p-sarah",
                            "document_id": "d2",
                            "title": "Sarah Example",
                            "snippet": (
                                "Her first charting hit appeared on Chart "
                                "Weekly's dance chart."
                            ),
                            "rank": 2,
                        },
                        {
                            "passage_id": "p-weekly",
                            "document_id": "d3",
                            "title": "Chart Weekly (magazine)",
                            "snippet": (
                                "Chart Weekly magazine first published its "
                                "American hit chart."
                            ),
                            "rank": 3,
                        },
                        {
                            "passage_id": "p-patti",
                            "document_id": "d4",
                            "title": "Patti Example",
                            "snippet": (
                                "Her first hit topped a Chart Weekly music chart."
                            ),
                            "rank": 4,
                        },
                    ],
                },
            }
        ]
        adapter = QARetrievalReactExecutionAdapter(
            gateway=SimpleNamespace(generate=lambda request: None),
            tool_registry=build_qa_tool_registry(FakeIndex()),
            max_turns=16,
            max_tool_calls=12,
            task_type="factual_qa",
            completion_policy="required_evidence",
        )
        schema = adapter._state_conditioned_response_schema(request, observations)
        assert schema is not None
        self.assertIn("oneOf", schema)
        branches = {
            branch["properties"]["name"]["const"]: branch
            for branch in schema["oneOf"]
        }
        self.assertEqual({"read", "search"}, set(branches))
        read_ids = branches["read"]["properties"]["arguments"]["properties"][
            "passage_id"
        ]["enum"]
        self.assertEqual({"p-hit", "p-weekly"}, set(read_ids))
        self.assertEqual(
            10,
            branches["search"]["properties"]["arguments"]["properties"][
                "limit"
            ]["const"],
        )
        false_positive = StructuredAction(
            ActionKind.TOOL,
            "read",
            {"passage_id": "p-sarah"},
            resource_id=QA_RETRIEVAL_TOOL_ID,
        )
        self.assertEqual(
            "qa_read_passage_id_not_from_search",
            adapter._tool_action_error(
                request=request,
                action=false_positive,
                observations=observations,
            ),
        )
        serialized = json.dumps(schema).casefold()
        for hidden_field in (
            "accepted_answers",
            "gold_answer",
            "evaluator",
            "final_answer",
        ):
            self.assertNotIn(hidden_field, serialized)

        false_only = json.loads(json.dumps(observations))
        false_only[0]["result"]["passage_ids"] = ["p-sarah", "p-patti"]
        false_only[0]["result"]["hits"] = [
            false_only[0]["result"]["hits"][1],
            false_only[0]["result"]["hits"][3],
        ]
        false_only_actions, false_only_completion = (
            adapter._state_conditioned_action_domain(request, false_only)
        )
        self.assertEqual(
            frozenset({(QA_RETRIEVAL_TOOL_ID, "search")}),
            false_only_actions,
        )
        self.assertFalse(false_only_completion)
        false_only_schema = adapter._state_conditioned_response_schema(
            request,
            false_only,
        )
        assert false_only_schema is not None
        self.assertEqual(
            "search",
            false_only_schema["properties"]["name"]["const"],
        )

    def test_model_continuation_consumes_read_action_but_preserves_public_state(
        self,
    ) -> None:
        search = {
            "observation_status": "success",
            "executed_action": {
                "kind": "tool",
                "name": "search",
                "resource_id": QA_RETRIEVAL_TOOL_ID,
                "arguments": {"query": "public entity relation", "limit": 5},
            },
            "result": {
                "operation": "search",
                "query": "public entity relation",
                "top_k": 5,
                "passage_ids": ["p1", "p2"],
            },
        }
        read = {
            "observation_status": "success",
            "executed_action": {
                "kind": "tool",
                "name": "read",
                "resource_id": QA_RETRIEVAL_TOOL_ID,
                "arguments": {"passage_id": "p1"},
            },
            "result": {
                "operation": "read",
                "passage_id": "p1",
                "passage": {
                    "title": "Public entity",
                    "text": "Public entity has the requested relation.",
                },
            },
        }
        observations = [search, read]
        visible = QARetrievalReactExecutionAdapter._model_visible_observations(
            observations
        )
        self.assertIn("executed_action", observations[1])
        self.assertNotIn("executed_action", visible[1])
        self.assertEqual("p1", visible[1]["result"]["passage_id"])

        adapter = QARetrievalReactExecutionAdapter(
            gateway=SimpleNamespace(generate=lambda request: None),
            tool_registry=build_qa_tool_registry(FakeIndex()),
            max_turns=8,
            max_tool_calls=12,
            task_type="factual_qa",
            completion_policy="required_evidence",
        )
        request = AgentRequest(
            request_id="trivia:consumed-read",
            run_id="trivia",
            graph_revision=1,
            problem="Which public entity has the requested relation?",
            agent=AgentNode(
                "retriever",
                "model",
                "retrieve evidence",
                role_family="evidence_retriever",
                allowed_tools=(QA_RETRIEVAL_TOOL_ID,),
                execution_mode="react",
            ),
            model=ModelSpec("model", "provider"),
            provider=ProviderSpec("provider", kind="test"),
            phase=ExecutionPhase.SINGLE,
            semantic_protocol=QA_VERIFIED_ANSWER_LINEAGE_PROTOCOL,
        )
        state = adapter._required_evidence_state(request, observations)
        self.assertEqual(("p1",), state.read_passage_ids)
        self.assertEqual(("p2",), state.latest_unread_passage_ids)

    async def test_unified_factual_completion_accepts_one_grounded_proposition(
        self,
    ) -> None:
        class FactualIndex(FakeIndex):
            def read(self, passage_id: str) -> FakePassage:
                self.read_calls.append(passage_id)
                return FakePassage(
                    passage_id,
                    "d1",
                    "France",
                    "Paris is the capital of France.",
                )

        artifact = {
            "question_scope": "What is the capital of France?",
            "answer_slot": {
                "answer_type": "short_answer",
                "answer_cardinality": "single",
                "qualifiers": [],
                "proposition_index": 0,
                "answer_field": "object_or_attribute_value",
            },
            "evidence_propositions": [
                {
                    "subject": "France",
                    "relation": "capital",
                    "object_or_attribute_value": "Paris",
                    "qualifiers": [],
                    "evidence_span": "Paris is the capital of France.",
                }
            ],
            "multi_hop_chain": ["France --capital--> Paris"],
            "candidate_answer": "Paris",
            "evidence": ["Paris is the capital of France."],
        }

        def action(name: str, arguments: object) -> str:
            return json.dumps(
                {
                    "arguments": arguments,
                    "kind": "complete" if name == "complete" else "tool",
                    "name": name,
                    "resource_id": (
                        None if name == "complete" else QA_RETRIEVAL_TOOL_ID
                    ),
                    "skill_id": None,
                }
            )

        class SequenceGateway:
            def __init__(self) -> None:
                self.outputs = [
                    action("search", {"query": "France capital", "limit": 5}),
                    action("read", {"passage_id": "p1"}),
                    action("complete", {"value": artifact}),
                ]

            async def generate(self, request: AgentRequest) -> AgentResponse:
                del request
                return AgentResponse(self.outputs.pop(0))

        index = FactualIndex()
        request = AgentRequest(
            request_id="trivia:one-proposition",
            run_id="trivia",
            graph_revision=1,
            problem="What is the capital of France?",
            agent=AgentNode(
                "reasoner",
                "model",
                "retrieve and bind the factual answer",
                role_family="reasoner",
                allowed_tools=(QA_RETRIEVAL_TOOL_ID,),
                execution_mode="react",
            ),
            model=ModelSpec("model", "provider"),
            provider=ProviderSpec("provider", kind="test"),
            phase=ExecutionPhase.SINGLE,
            semantic_protocol=QA_VERIFIED_ANSWER_LINEAGE_PROTOCOL,
        )

        response = await QARetrievalReactExecutionAdapter(
            gateway=SequenceGateway(),
            tool_registry=build_qa_tool_registry(index),
            max_turns=3,
            max_tool_calls=2,
            task_type="factual_qa",
            completion_policy="required_evidence",
        ).execute(request)

        self.assertEqual(
            json.dumps(
                artifact,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ),
            response.text,
        )
        self.assertEqual([("France capital", 5)], index.search_calls)
        self.assertEqual(["p1"], index.read_calls)

    def test_unified_factual_evidence_rejection_advances_retrieval_strategy(
        self,
    ) -> None:
        adapter = QARetrievalReactExecutionAdapter(
            gateway=SimpleNamespace(generate=lambda request: None),
            tool_registry=build_qa_tool_registry(FakeIndex()),
            max_turns=8,
            max_tool_calls=10,
            task_type="factual_qa",
            completion_policy="required_evidence",
        )
        request = AgentRequest(
            request_id="trivia:retry",
            run_id="trivia",
            graph_revision=1,
            problem="Who wrote the novel?",
            agent=AgentNode(
                "reasoner",
                "model",
                "retrieve evidence and bind the requested relation",
                role_family="reasoner",
                allowed_tools=(QA_RETRIEVAL_TOOL_ID,),
                execution_mode="react",
            ),
            model=ModelSpec("model", "provider"),
            provider=ProviderSpec("provider", kind="test"),
            phase=ExecutionPhase.SINGLE,
            semantic_protocol=QA_VERIFIED_ANSWER_LINEAGE_PROTOCOL,
        )
        search = {
            "observation_status": "success",
            "executed_action": {
                "kind": "tool",
                "name": "search",
                "resource_id": QA_RETRIEVAL_TOOL_ID,
                "skill_id": None,
                "arguments": {"query": "Novel author", "limit": 5},
            },
            "result": {
                "operation": "search",
                "query": "Novel author",
                "top_k": 5,
                "passage_ids": ["p1"],
            },
        }
        read = {
            "observation_status": "success",
            "executed_action": {
                "kind": "tool",
                "name": "read",
                "resource_id": QA_RETRIEVAL_TOOL_ID,
                "skill_id": None,
                "arguments": {"passage_id": "p1"},
            },
            "result": {
                "operation": "read",
                "passage_id": "p1",
                "passage": {"text": "A different entity is discussed here."},
            },
        }
        rejection = {
            "observation_status": "schema_invalid",
            "public_error_code": (
                "qa_semantic_evidence_provenance_invalid: target relation absent"
            ),
        }
        observations = [search, read, rejection]

        admitted, completion = adapter._state_conditioned_action_domain(
            request,
            observations,
        )
        self.assertEqual(
            frozenset({(QA_RETRIEVAL_TOOL_ID, "search")}), admitted
        )
        self.assertFalse(completion)
        retry_schema = adapter._state_conditioned_response_schema(
            request,
            observations,
        )
        assert retry_schema is not None
        self.assertEqual(
            10,
            retry_schema["properties"]["arguments"]["properties"]
            ["limit"]["const"],
        )
        retry_contract = adapter._contract(request, observations)
        self.assertIn("`spelling_normalization`", retry_contract)
        self.assertIn('"novel author"', retry_contract)

        expanded_same_query = StructuredAction(
            ActionKind.TOOL,
            "search",
            {"query": "  NOVEL   AUTHOR ", "limit": 10},
            resource_id=QA_RETRIEVAL_TOOL_ID,
        )
        self.assertIsNone(
            adapter._tool_action_error(
                request=request,
                action=expanded_same_query,
                observations=observations,
            ),
        )
        expanded_search = {
            "observation_status": "success",
            "executed_action": {
                "kind": "tool",
                "name": "search",
                "resource_id": QA_RETRIEVAL_TOOL_ID,
                "skill_id": None,
                "arguments": {"query": "  NOVEL   AUTHOR ", "limit": 10},
            },
            "result": {
                "operation": "search",
                "query": "NOVEL AUTHOR",
                "top_k": 10,
                "passage_ids": [],
            },
        }
        expanded_state = adapter._required_evidence_state(
            request,
            [*observations, expanded_search],
        )
        self.assertEqual(2, expanded_state.search_attempt_count)
        self.assertEqual(1, expanded_state.strategy_progress_count)
        self.assertEqual(1, expanded_state.recall_expansion_count)
        expanded_schema = adapter._state_conditioned_response_schema(
            request,
            [*observations, expanded_search],
        )
        assert expanded_schema is not None
        self.assertEqual(
            10,
            expanded_schema["properties"]["arguments"]["properties"]
            ["limit"]["const"],
        )
        self.assertIn(
            "spelling_normalization",
            adapter._contract(request, [*observations, expanded_search]),
        )
        repeated_expansion = StructuredAction(
            ActionKind.TOOL,
            "search",
            {"query": "novel author", "limit": 10},
            resource_id=QA_RETRIEVAL_TOOL_ID,
        )
        self.assertEqual(
            "qa_retrieval_duplicate_normalized_query",
            adapter._tool_action_error(
                request=request,
                action=repeated_expansion,
                observations=[*observations, expanded_search],
            ),
        )
        repeated_pair = StructuredAction(
            ActionKind.TOOL,
            "search",
            {"query": "  NOVEL   AUTHOR ", "limit": 5},
            resource_id=QA_RETRIEVAL_TOOL_ID,
        )
        self.assertEqual(
            "qa_retrieval_duplicate_normalized_query",
            adapter._tool_action_error(
                request=request,
                action=repeated_pair,
                observations=observations,
            ),
        )

        retry_search = {
            "observation_status": "success",
            "executed_action": {
                "kind": "tool",
                "name": "search",
                "resource_id": QA_RETRIEVAL_TOOL_ID,
                "skill_id": None,
                "arguments": {"query": "Novelist pen name", "limit": 10},
            },
            "result": {
                "operation": "search",
                "query": "Novelist pen name",
                "top_k": 10,
                "passage_ids": ["p2"],
            },
        }
        admitted_after_retry, completion_after_retry = (
            adapter._state_conditioned_action_domain(
                request,
                [*observations, retry_search],
            )
        )
        self.assertEqual(
            frozenset(
                {
                    (QA_RETRIEVAL_TOOL_ID, "read"),
                    (QA_RETRIEVAL_TOOL_ID, "search"),
                }
            ),
            admitted_after_retry,
        )
        self.assertFalse(completion_after_retry)

    def test_trivia_query_rewrite_rejects_question_external_date_candidate(
        self,
    ) -> None:
        adapter = QARetrievalReactExecutionAdapter(
            gateway=SimpleNamespace(generate=lambda request: None),
            tool_registry=build_qa_tool_registry(FakeIndex()),
            max_turns=16,
            max_tool_calls=12,
            task_type="factual_qa",
            completion_policy="required_evidence",
        )
        question = (
            "In which decade did Billboard magazine first publish an American "
            "hit chart?"
        )
        request = AgentRequest(
            request_id="trivia:query-candidate-gate",
            run_id="trivia",
            graph_revision=1,
            problem=question,
            agent=AgentNode(
                "retriever",
                "model",
                "retrieve the original entity and relation",
                role_family="evidence_retriever",
                allowed_tools=(QA_RETRIEVAL_TOOL_ID,),
                execution_mode="react",
            ),
            model=ModelSpec("model", "provider"),
            provider=ProviderSpec("provider", kind="test"),
            phase=ExecutionPhase.SINGLE,
            semantic_protocol=QA_VERIFIED_ANSWER_LINEAGE_PROTOCOL,
        )
        observations = [
            {
                "observation_status": "success",
                "executed_action": {
                    "kind": "tool",
                    "name": "search",
                    "resource_id": QA_RETRIEVAL_TOOL_ID,
                    "arguments": {
                        "query": "Billboard first American hit chart",
                        "limit": 5,
                    },
                },
                "result": {
                    "operation": "search",
                    "query": "Billboard first American hit chart",
                    "top_k": 5,
                    "passage_ids": ["p1"],
                },
            },
            {
                "observation_status": "success",
                "executed_action": {
                    "kind": "tool",
                    "name": "read",
                    "resource_id": QA_RETRIEVAL_TOOL_ID,
                    "arguments": {"passage_id": "p1"},
                },
                "result": {
                    "operation": "read",
                    "passage_id": "p1",
                    "passage": {"text": "Irrelevant chart evidence."},
                },
            },
            {
                "observation_status": "schema_invalid",
                "public_error_code": (
                    "qa_semantic_evidence_provenance_invalid: relation mismatch"
                ),
            },
        ]
        narrowed = StructuredAction(
            ActionKind.TOOL,
            "search",
            {
                "query": (
                    "Billboard magazine first publish hit chart 1930s"
                ),
                "limit": 10,
            },
            resource_id=QA_RETRIEVAL_TOOL_ID,
        )
        self.assertEqual(
            "qa_retrieval_query_candidate_answer_injection",
            adapter._tool_action_error(
                request=request,
                action=narrowed,
                observations=observations,
            ),
        )
        neutral = StructuredAction(
            ActionKind.TOOL,
            "search",
            {
                "query": (
                    "Billboard magazine first publish music hit parade"
                ),
                "limit": 10,
            },
            resource_id=QA_RETRIEVAL_TOOL_ID,
        )
        self.assertIsNone(
            adapter._tool_action_error(
                request=request,
                action=neutral,
                observations=observations,
            )
        )
        self.assertIn(
            "Remove every question-external numeric, year, or decade candidate",
            adapter._public_semantic_repair_instruction(
                "qa_retrieval_query_candidate_answer_injection"
            ),
        )

    def test_recall_expansion_budget_still_admits_all_five_strategies(
        self,
    ) -> None:
        adapter = QARetrievalReactExecutionAdapter(
            gateway=SimpleNamespace(generate=lambda request: None),
            tool_registry=build_qa_tool_registry(FakeIndex()),
            max_turns=16,
            max_tool_calls=12,
            task_type="factual_qa",
            completion_policy="required_evidence",
        )
        request = AgentRequest(
            request_id="trivia:strategy-budget",
            run_id="trivia",
            graph_revision=1,
            problem=(
                "In which decade did Billboard magazine first publish an "
                "American hit chart?"
            ),
            agent=AgentNode(
                "retriever",
                "model",
                "retrieve the original entity and relation",
                role_family="evidence_retriever",
                allowed_tools=(QA_RETRIEVAL_TOOL_ID,),
                execution_mode="react",
            ),
            model=ModelSpec("model", "provider"),
            provider=ProviderSpec("provider", kind="test"),
            phase=ExecutionPhase.SINGLE,
            semantic_protocol=QA_VERIFIED_ANSWER_LINEAGE_PROTOCOL,
        )

        observations: list[dict[str, object]] = []

        def append_cycle(query: str, limit: int, passage_id: str) -> None:
            observations.extend(
                [
                    {
                        "observation_status": "success",
                        "executed_action": {
                            "kind": "tool",
                            "name": "search",
                            "resource_id": QA_RETRIEVAL_TOOL_ID,
                            "arguments": {"query": query, "limit": limit},
                        },
                        "result": {
                            "operation": "search",
                            "query": query,
                            "top_k": limit,
                            "passage_ids": [passage_id],
                        },
                    },
                    {
                        "observation_status": "success",
                        "executed_action": {
                            "kind": "tool",
                            "name": "read",
                            "resource_id": QA_RETRIEVAL_TOOL_ID,
                            "arguments": {"passage_id": passage_id},
                        },
                        "result": {
                            "operation": "read",
                            "passage_id": passage_id,
                            "passage": {"text": "Public but insufficient evidence."},
                        },
                    },
                    {
                        "observation_status": "schema_invalid",
                        "public_error_code": (
                            "qa_semantic_evidence_provenance_invalid: relation mismatch"
                        ),
                    },
                ]
            )

        append_cycle(
            "Billboard magazine first publish American hit chart", 5, "p1"
        )
        append_cycle(
            "Billboard magazine first publish American hit chart", 10, "p2"
        )
        append_cycle(
            "Billboard magazine first published American hit chart", 10, "p3"
        )
        append_cycle(
            "Billboard magazine first published American music hit parade",
            15,
            "p4",
        )
        append_cycle(
            "Billboard magazine publication first American music hit parade",
            20,
            "p5",
        )

        final_strategy = StructuredAction(
            ActionKind.TOOL,
            "search",
            {
                "query": (
                    "Billboard magazine published first American music hit parade history"
                ),
                "limit": 25,
            },
            resource_id=QA_RETRIEVAL_TOOL_ID,
        )
        self.assertIsNone(
            adapter._tool_action_error(
                request=request,
                action=final_strategy,
                observations=observations,
            )
        )
        observations.pop()
        append_cycle(
            "Billboard magazine published first American music hit parade history",
            25,
            "p6",
        )
        observations.pop()
        state = adapter._required_evidence_state(request, observations)
        self.assertEqual(6, state.search_attempt_count)
        self.assertEqual(5, state.strategy_progress_count)
        self.assertEqual(1, state.recall_expansion_count)
        self.assertEqual(12, state.dispatched_tool_calls)
        admitted, completion = adapter._state_conditioned_action_domain(
            request,
            observations,
        )
        self.assertEqual(frozenset(), admitted)
        self.assertTrue(completion)

    def test_unified_factual_upstream_read_does_not_mask_evidence_recovery(
        self,
    ) -> None:
        adapter = QARetrievalReactExecutionAdapter(
            gateway=SimpleNamespace(generate=lambda request: None),
            tool_registry=build_qa_tool_registry(FakeIndex()),
            max_turns=8,
            max_tool_calls=10,
            task_type="factual_qa",
            completion_policy="required_evidence",
        )
        upstream_read_receipt = {
            "tool_id": QA_RETRIEVAL_TOOL_ID,
            "request": {
                "action": "read",
                "arguments": {"passage_id": "upstream-p1"},
            },
            "result": {
                "value": {
                    "operation": "read",
                    "passage_id": "upstream-p1",
                    "passage": {
                        "passage_id": "upstream-p1",
                        "text": "This passage discusses a different entity.",
                    },
                },
                "completed": True,
            },
            "error_type": None,
        }
        request = AgentRequest(
            request_id="trivia:upstream-recovery",
            run_id="trivia",
            graph_revision=2,
            problem="Where was the actor born?",
            agent=AgentNode(
                "reasoner",
                "model",
                "bind retrieved evidence to the requested relation",
                role_family="reasoner",
                allowed_tools=(QA_RETRIEVAL_TOOL_ID,),
                execution_mode="react",
            ),
            model=ModelSpec("model", "provider"),
            provider=ProviderSpec("provider", kind="test"),
            phase=ExecutionPhase.SINGLE,
            semantic_protocol=QA_VERIFIED_ANSWER_LINEAGE_PROTOCOL,
            upstream=(
                UpstreamMessage(
                    "retriever",
                    "reasoner",
                    "retrieved upstream-p1",
                    graph_revision=2,
                    artifact_type="retrieval_evidence",
                    tool_receipts=(upstream_read_receipt,),
                ),
            ),
        )

        initial_actions, initial_completion = (
            adapter._state_conditioned_action_domain(request, [])
        )
        self.assertEqual(
            frozenset({(QA_RETRIEVAL_TOOL_ID, "search")}),
            initial_actions,
        )
        self.assertFalse(initial_completion)

        rejection = {
            "observation_status": "schema_invalid",
            "public_error_code": (
                "qa_semantic_evidence_provenance_invalid: target relation absent"
            ),
        }
        recovery_actions, recovery_completion = (
            adapter._state_conditioned_action_domain(request, [rejection])
        )
        self.assertEqual(
            frozenset({(QA_RETRIEVAL_TOOL_ID, "search")}),
            recovery_actions,
        )
        self.assertFalse(recovery_completion)

        own_search = {
            "observation_status": "success",
            "executed_action": {
                "kind": "tool",
                "name": "search",
                "resource_id": QA_RETRIEVAL_TOOL_ID,
                "skill_id": None,
                "arguments": {"query": "actor birthplace", "limit": 5},
            },
            "result": {
                "operation": "search",
                "query": "actor birthplace",
                "top_k": 5,
                "passage_ids": ["own-p1"],
            },
        }
        own_read = {
            "observation_status": "success",
            "executed_action": {
                "kind": "tool",
                "name": "read",
                "resource_id": QA_RETRIEVAL_TOOL_ID,
                "skill_id": None,
                "arguments": {"passage_id": "own-p1"},
            },
            "result": {
                "operation": "read",
                "passage_id": "own-p1",
                "passage": {"text": "The actor was born in the city."},
            },
        }
        final_actions, final_completion = (
            adapter._state_conditioned_action_domain(
                request,
                [rejection, own_search, own_read],
            )
        )
        self.assertEqual(frozenset(), final_actions)
        self.assertTrue(final_completion)

    def test_unified_factual_reasoner_admits_only_valid_upstream_evidence_artifacts(
        self,
    ) -> None:
        adapter = QARetrievalReactExecutionAdapter(
            gateway=SimpleNamespace(generate=lambda request: None),
            tool_registry=build_qa_tool_registry(FakeIndex()),
            max_turns=8,
            max_tool_calls=10,
            task_type="factual_qa",
            completion_policy="required_evidence",
        )
        question = "Which city does David Soul come from?"
        evidence = "Soul comes from Chicago, Illinois."
        artifact = json.dumps(
            {
                "question_scope": question,
                "entity_identity": {
                    "question_surface": "David Soul",
                    "evidence_surface": "Soul",
                },
                "target_relation": "comes from",
                "answer_type_constraint": "entity",
                "evidence_proposition": {
                    "subject": "Soul",
                    "predicate": "comes from",
                    "object_or_attribute_value": "Chicago, Illinois",
                },
                "evidence_span": evidence,
                "passage_id": "david-soul",
            }
        )
        receipt = {
            "tool_id": QA_RETRIEVAL_TOOL_ID,
            "tool_version": "frozen-index-v1",
            "request": {
                "action": "read",
                "arguments": {"passage_id": "david-soul"},
            },
            "result": {
                "value": {
                    "operation": "read",
                    "passage_id": "david-soul",
                    "passage": {
                        "passage_id": "david-soul",
                        "title": "David Soul",
                        "text": evidence,
                    },
                },
                "completed": True,
            },
            "error_type": None,
        }
        valid_message = UpstreamMessage(
            "retriever-valid",
            "reasoner",
            artifact,
            graph_revision=2,
            artifact_type="retrieval_evidence",
            tool_receipts=(receipt,),
        )
        invalid_message = UpstreamMessage(
            "retriever-invalid",
            "reasoner",
            "A read completed, but this is not a grounded evidence artifact.",
            graph_revision=2,
            artifact_type="retrieval_evidence",
            tool_receipts=(receipt,),
        )
        base_request = AgentRequest(
            request_id="trivia:validated-upstream-evidence",
            run_id="trivia",
            graph_revision=2,
            problem=question,
            agent=AgentNode(
                "reasoner",
                "model",
                "bind retrieved evidence to the requested relation",
                role_family="reasoner",
                allowed_tools=(QA_RETRIEVAL_TOOL_ID,),
                execution_mode="react",
            ),
            model=ModelSpec("model", "provider"),
            provider=ProviderSpec("provider", kind="test"),
            phase=ExecutionPhase.SINGLE,
            semantic_protocol=QA_VERIFIED_ANSWER_LINEAGE_PROTOCOL,
        )

        cases = (
            (
                "direct predecessor",
                replace(base_request, upstream=(valid_message,)),
            ),
            (
                "peer draft",
                replace(
                    base_request,
                    phase=ExecutionPhase.REVISION,
                    peer_draft=valid_message,
                ),
            ),
            (
                "any valid predecessor",
                replace(
                    base_request,
                    upstream=(invalid_message, valid_message),
                ),
            ),
        )
        for label, request in cases:
            with self.subTest(label=label):
                admitted_receipts = (
                    adapter._validated_upstream_evidence_receipts(request)
                )
                self.assertEqual(1, len(admitted_receipts))
                actions, completion = adapter._state_conditioned_action_domain(
                    request,
                    [],
                )
                self.assertEqual(
                    frozenset({(QA_RETRIEVAL_TOOL_ID, "search")}),
                    actions,
                )
                self.assertTrue(completion)

        invalid_only = replace(base_request, upstream=(invalid_message,))
        self.assertEqual(
            (),
            adapter._validated_upstream_evidence_receipts(invalid_only),
        )
        _, invalid_completion = adapter._state_conditioned_action_domain(
            invalid_only,
            [],
        )
        self.assertFalse(invalid_completion)

    def test_hotpot_upstream_read_completion_admission_is_unchanged(self) -> None:
        receipt = {
            "tool_id": QA_RETRIEVAL_TOOL_ID,
            "request": {
                "action": "read",
                "arguments": {"passage_id": "hotpot-p1"},
            },
            "result": {
                "value": {
                    "operation": "read",
                    "passage_id": "hotpot-p1",
                    "passage": {
                        "passage_id": "hotpot-p1",
                        "text": "A successful HotpotQA predecessor read.",
                    },
                },
                "completed": True,
            },
            "error_type": None,
        }
        request = AgentRequest(
            request_id="hotpot:upstream-admission-regression",
            run_id="hotpot",
            graph_revision=1,
            problem="Which person is described by both passages?",
            agent=AgentNode(
                "reasoner",
                "model",
                "bind multi-hop evidence",
                role_family="reasoner",
                allowed_tools=(QA_RETRIEVAL_TOOL_ID,),
                execution_mode="react",
            ),
            model=ModelSpec("model", "provider"),
            provider=ProviderSpec("provider", kind="test"),
            phase=ExecutionPhase.SINGLE,
            semantic_protocol="hotpotqa_verified_answer_slot_v1",
            upstream=(
                UpstreamMessage(
                    "retriever",
                    "reasoner",
                    "legacy HotpotQA predecessor artifact",
                    graph_revision=1,
                    artifact_type="retrieval_evidence",
                    tool_receipts=(receipt,),
                ),
            ),
        )
        adapter = QARetrievalReactExecutionAdapter(
            gateway=SimpleNamespace(generate=lambda request: None),
            tool_registry=build_qa_tool_registry(FakeIndex()),
            max_turns=8,
            max_tool_calls=10,
            task_type="multi_hop_qa",
            completion_policy="required_evidence",
        )

        _, completion = adapter._state_conditioned_action_domain(request, [])

        self.assertTrue(completion)

    def test_unified_factual_evidence_rejection_can_read_unread_hit_or_search(
        self,
    ) -> None:
        adapter = QARetrievalReactExecutionAdapter(
            gateway=SimpleNamespace(generate=lambda request: None),
            tool_registry=build_qa_tool_registry(FakeIndex()),
            max_turns=8,
            max_tool_calls=8,
            task_type="factual_qa",
            completion_policy="required_evidence",
        )
        request = AgentRequest(
            request_id="trivia:unread-hit-after-rejection",
            run_id="trivia",
            graph_revision=1,
            problem=(
                "In which decade did Billboard magazine first publish an "
                "American hit chart?"
            ),
            agent=AgentNode(
                "retriever",
                "model",
                "retrieve question-grounded evidence",
                role_family="evidence_retriever",
                allowed_tools=(QA_RETRIEVAL_TOOL_ID,),
                execution_mode="react",
            ),
            model=ModelSpec("model", "provider"),
            provider=ProviderSpec("provider", kind="test"),
            phase=ExecutionPhase.SINGLE,
            semantic_protocol=QA_VERIFIED_ANSWER_LINEAGE_PROTOCOL,
        )
        observations = [
            {
                "observation_status": "success",
                "executed_action": {
                    "kind": "tool",
                    "name": "search",
                    "resource_id": QA_RETRIEVAL_TOOL_ID,
                    "skill_id": None,
                    "arguments": {
                        "query": "Billboard first American hit chart",
                        "limit": 5,
                    },
                },
                "result": {
                    "operation": "search",
                    "query": "Billboard first American hit chart",
                    "top_k": 5,
                    "passage_ids": ["p1", "p2"],
                },
            },
            {
                "observation_status": "success",
                "executed_action": {
                    "kind": "tool",
                    "name": "read",
                    "resource_id": QA_RETRIEVAL_TOOL_ID,
                    "skill_id": None,
                    "arguments": {"passage_id": "p1"},
                },
                "result": {
                    "operation": "read",
                    "passage_id": "p1",
                    "passage": {
                        "passage_id": "p1",
                        "text": "This passage discusses an unrelated chart.",
                    },
                },
            },
            {
                "observation_status": "schema_invalid",
                "public_error_code": (
                    "qa_semantic_evidence_provenance_invalid: "
                    "target relation absent"
                ),
            },
        ]

        state = adapter._required_evidence_state(request, observations)
        actions, completion = adapter._state_conditioned_action_domain(
            request,
            observations,
        )

        self.assertEqual(("p2",), state.latest_unread_passage_ids)
        self.assertEqual(
            frozenset(
                {
                    (QA_RETRIEVAL_TOOL_ID, "read"),
                    (QA_RETRIEVAL_TOOL_ID, "search"),
                }
            ),
            actions,
        )
        self.assertFalse(completion)

    async def test_unverified_factual_strategy_exhaustion_is_not_coverage(
        self,
    ) -> None:
        class EmptyIndex(FakeIndex):
            def search(self, query: str, *, limit: int) -> tuple[FakeHit, ...]:
                self.search_calls.append((query, limit))
                return ()

        def search_action(query: str, limit: int) -> str:
            return json.dumps(
                {
                    "arguments": {"query": query, "limit": limit},
                    "kind": "tool",
                    "name": "search",
                    "resource_id": QA_RETRIEVAL_TOOL_ID,
                    "skill_id": None,
                }
            )

        class SequenceGateway:
            def __init__(self) -> None:
                self.outputs = [
                    search_action("Entity relation", 5),
                    search_action("Entitiy relation", 10),
                    search_action("Officeholder relation", 15),
                    search_action("Entity public office position", 20),
                    search_action("Person held office", 25),
                    json.dumps(
                        {
                            "arguments": {"value": "unsupported guess"},
                            "kind": "complete",
                            "name": "complete",
                            "resource_id": None,
                            "skill_id": None,
                        }
                    ),
                ]

            async def generate(self, request: AgentRequest) -> AgentResponse:
                del request
                return AgentResponse(self.outputs.pop(0))

        request = AgentRequest(
            request_id="trivia:coverage",
            run_id="trivia",
            graph_revision=1,
            problem="Who held the office?",
            agent=AgentNode(
                "reasoner",
                "model",
                "retrieve evidence before answering",
                role_family="reasoner",
                allowed_tools=(QA_RETRIEVAL_TOOL_ID,),
                execution_mode="react",
            ),
            model=ModelSpec("model", "provider"),
            provider=ProviderSpec("provider", kind="test"),
            phase=ExecutionPhase.SINGLE,
            semantic_protocol=QA_VERIFIED_ANSWER_LINEAGE_PROTOCOL,
        )

        with self.assertRaises(ReactExecutionError) as caught:
            await QARetrievalReactExecutionAdapter(
                gateway=SequenceGateway(),
                tool_registry=build_qa_tool_registry(EmptyIndex()),
                max_turns=6,
                max_tool_calls=10,
                task_type="factual_qa",
                completion_policy="required_evidence",
            ).execute(request)

        error = caught.exception
        self.assertEqual(5, len(error.tool_receipts))
        self.assertIs(True, error.tool_plan_exhausted)
        self.assertEqual(
            "retrieval_strategy_failure",
            error.react_trace[-1]["public_error_code"],
        )
        terminal_diagnosis = error.react_trace[-1][
            "terminal_failure_diagnosis"
        ]
        self.assertIs(True, terminal_diagnosis["tool_plan_exhausted"])
        self.assertEqual(5, terminal_diagnosis["retrieval_attempt_count"])
        self.assertEqual(
            [5, 10, 15, 20, 25],
            terminal_diagnosis["search_top_ks"],
        )
        self.assertEqual(
            [
                "initial_retrieval",
                "spelling_normalization",
                "alias_expansion",
                "entity_disambiguation",
                "query_rewriting",
            ],
            terminal_diagnosis["retrieval_strategy_schedule_prefix"],
        )
        self.assertFalse(terminal_diagnosis["strategy_semantics_verified"])
        self.assertNotIn(
            "retrieval_strategies_attempted",
            terminal_diagnosis,
        )

    async def test_same_query_top_k_expansion_does_not_exhaust_strategies(
        self,
    ) -> None:
        class EmptyIndex(FakeIndex):
            def search(self, query: str, *, limit: int) -> tuple[FakeHit, ...]:
                self.search_calls.append((query, limit))
                return ()

        def search_action(query: str, limit: int) -> str:
            return json.dumps(
                {
                    "arguments": {"query": query, "limit": limit},
                    "kind": "tool",
                    "name": "search",
                    "resource_id": QA_RETRIEVAL_TOOL_ID,
                    "skill_id": None,
                }
            )

        class SequenceGateway:
            def __init__(self) -> None:
                self.requests: list[AgentRequest] = []
                self.outputs = [
                    search_action(
                        "Billboard magazine first publish hit chart decade", 5
                    ),
                    search_action(
                        "BILLBOARD MAGAZINE first publish hit chart decade", 10
                    ),
                    search_action(
                        "Billboard magazine first publish hit chart decade", 10
                    ),
                    search_action(
                        "Billboard magazine first publish hit chart decade", 10
                    ),
                ]

            async def generate(self, request: AgentRequest) -> AgentResponse:
                self.requests.append(request)
                return AgentResponse(self.outputs.pop(0))

        index = EmptyIndex()
        request = AgentRequest(
            request_id="trivia:same-query-expansion",
            run_id="trivia",
            graph_revision=1,
            problem=(
                "In which decade did Billboard magazine first publish an "
                "American hit chart?"
            ),
            agent=AgentNode(
                "reasoner",
                "model",
                "retrieve evidence before answering",
                role_family="reasoner",
                allowed_tools=(QA_RETRIEVAL_TOOL_ID,),
                execution_mode="react",
            ),
            model=ModelSpec("model", "provider"),
            provider=ProviderSpec("provider", kind="test"),
            phase=ExecutionPhase.SINGLE,
            semantic_protocol=QA_VERIFIED_ANSWER_LINEAGE_PROTOCOL,
        )

        gateway = SequenceGateway()
        with self.assertRaises(ReactExecutionError) as caught:
            await QARetrievalReactExecutionAdapter(
                gateway=gateway,
                tool_registry=build_qa_tool_registry(index),
                max_turns=4,
                max_tool_calls=10,
                task_type="factual_qa",
                completion_policy="required_evidence",
            ).execute(request)

        error = caught.exception
        self.assertEqual(
            [
                ("Billboard magazine first publish hit chart decade", 5),
                ("BILLBOARD MAGAZINE first publish hit chart decade", 10),
            ],
            index.search_calls,
        )
        self.assertEqual(2, len(error.tool_receipts))
        terminal_diagnosis = error.react_trace[-1][
            "terminal_failure_diagnosis"
        ]
        self.assertEqual(
            "retrieval_strategy_failure",
            terminal_diagnosis["public_error_code"],
        )
        self.assertEqual(2, terminal_diagnosis["retrieval_attempt_count"])
        self.assertEqual(
            1,
            terminal_diagnosis["retrieval_strategy_progress_count"],
        )
        self.assertEqual(1, terminal_diagnosis["recall_expansion_count"])
        self.assertEqual(
            ["initial_retrieval"],
            terminal_diagnosis["retrieval_strategy_schedule_prefix"],
        )
        self.assertFalse(
            terminal_diagnosis["normalized_query_novelty_verified"]
        )
        continuation_contract = gateway.requests[1].agent.contract
        self.assertIn(
            '"required_strategy":"spelling_normalization"',
            continuation_contract,
        )
        self.assertIn('"required_top_k":10', continuation_contract)
        self.assertIn(
            '"prior_normalized_queries":["billboard magazine first publish hit chart decade"]',
            continuation_contract,
        )

    async def test_unverified_factual_strategy_after_reads_is_not_coverage(
        self,
    ) -> None:
        class MismatchedIndex(FakeIndex):
            def search(self, query: str, *, limit: int) -> tuple[FakeHit, ...]:
                self.search_calls.append((query, limit))
                passage_id = f"p{len(self.search_calls)}"
                return (
                    FakeHit(
                        passage_id,
                        f"d{len(self.search_calls)}",
                        "Different entity",
                        "Evidence for another relation.",
                        1,
                    ),
                )

            def read(self, passage_id: str) -> FakePassage:
                self.read_calls.append(passage_id)
                return FakePassage(
                    passage_id,
                    passage_id,
                    "Different entity",
                    f"Different entity {passage_id} held another position.",
                )

        artifact = {
            "question_scope": "Who held the office?",
            "answer_slot": {
                "answer_type": "entity",
                "answer_cardinality": "single",
                "qualifiers": [],
                "proposition_index": 0,
                "answer_field": "subject",
            },
            "evidence_propositions": [
                {
                    "subject": "Unsupported Person",
                    "relation": "held",
                    "object_or_attribute_value": "the office",
                    "qualifiers": [],
                    "evidence_span": "Unsupported Person held the office.",
                }
            ],
            "multi_hop_chain": ["Unsupported Person --held--> the office"],
            "candidate_answer": "Unsupported Person",
            "evidence": ["Unsupported Person held the office."],
        }

        def action(name: str, arguments: object) -> str:
            return json.dumps(
                {
                    "arguments": arguments,
                    "kind": "complete" if name == "complete" else "tool",
                    "name": name,
                    "resource_id": (
                        None if name == "complete" else QA_RETRIEVAL_TOOL_ID
                    ),
                    "skill_id": None,
                }
            )

        queries = (
            "Office holder",
            "Offise holder",
            "Officeholder position",
            "Public official held office",
            "Person occupied office",
        )
        outputs: list[str] = []
        for index, (query, limit) in enumerate(
            zip(queries, (5, 10, 15, 20, 25)),
            start=1,
        ):
            outputs.extend(
                [
                    action("search", {"query": query, "limit": limit}),
                    action("read", {"passage_id": f"p{index}"}),
                    action("complete", {"value": artifact}),
                ]
            )
        # After the fifth provenance rejection the public action domain is
        # empty. Five lexically distinct queries do not establish that the
        # declared relation transformations actually ran, so this cannot be
        # classified as frozen-knowledge-base coverage failure.
        outputs.append(action("complete", {"value": artifact}))

        class SequenceGateway:
            def __init__(self) -> None:
                self.outputs = list(outputs)

            async def generate(self, request: AgentRequest) -> AgentResponse:
                del request
                return AgentResponse(self.outputs.pop(0))

        request = AgentRequest(
            request_id="trivia:coverage-after-reads",
            run_id="trivia",
            graph_revision=1,
            problem="Who held the office?",
            agent=AgentNode(
                "reasoner",
                "model",
                "retrieve evidence and bind the requested entity and relation",
                role_family="reasoner",
                allowed_tools=(QA_RETRIEVAL_TOOL_ID,),
                execution_mode="react",
            ),
            model=ModelSpec("model", "provider"),
            provider=ProviderSpec("provider", kind="test"),
            phase=ExecutionPhase.SINGLE,
            semantic_protocol=QA_VERIFIED_ANSWER_LINEAGE_PROTOCOL,
        )

        with self.assertRaises(ReactExecutionError) as caught:
            await QARetrievalReactExecutionAdapter(
                gateway=SequenceGateway(),
                tool_registry=build_qa_tool_registry(MismatchedIndex()),
                max_turns=16,
                max_tool_calls=10,
                task_type="factual_qa",
                completion_policy="required_evidence",
            ).execute(request)

        error = caught.exception
        self.assertEqual(10, len(error.tool_receipts))
        self.assertIs(True, error.tool_plan_exhausted)
        terminal_diagnosis = error.react_trace[-1][
            "terminal_failure_diagnosis"
        ]
        self.assertIs(True, terminal_diagnosis["tool_plan_exhausted"])
        self.assertEqual(
            "retrieval_strategy_failure",
            terminal_diagnosis["public_error_code"],
        )
        self.assertEqual(5, terminal_diagnosis["retrieval_attempt_count"])
        self.assertEqual(
            [5, 10, 15, 20, 25],
            terminal_diagnosis["search_top_ks"],
        )

    async def test_react_read_requires_canonical_id_from_successful_search(self) -> None:
        index = FakeIndex()
        registry = build_qa_tool_registry(index)

        def action(name: str, arguments: object) -> str:
            return json.dumps(
                {
                    "arguments": arguments,
                    "kind": "complete" if name == "complete" else "tool",
                    "name": name,
                    "resource_id": (
                        None if name == "complete" else QA_RETRIEVAL_TOOL_ID
                    ),
                    "skill_id": None,
                }
            )

        class SequenceGateway:
            def __init__(self) -> None:
                self.outputs = [
                    action("read", {"passage_id": "Ada Lovelace"}),
                    action("search", {"query": "Ada Lovelace", "limit": 1}),
                    action("read", {"passage_id": "wrong-id"}),
                    action("read", {"passage_id": "p1"}),
                    action("complete", {"value": "Ada Lovelace"}),
                ]

            async def generate(self, request: AgentRequest) -> AgentResponse:
                del request
                return AgentResponse(self.outputs.pop(0))

        request = AgentRequest(
            request_id="qa:react",
            run_id="qa",
            graph_revision=1,
            problem="Who wrote the first published algorithm?",
            agent=AgentNode(
                "retriever",
                "model",
                "retrieve one public passage and answer",
                allowed_tools=(QA_RETRIEVAL_TOOL_ID,),
                execution_mode="react",
            ),
            model=ModelSpec("model", "provider"),
            provider=ProviderSpec("provider", kind="test"),
            phase=ExecutionPhase.SINGLE,
        )
        response = await QARetrievalReactExecutionAdapter(
            gateway=SequenceGateway(),
            tool_registry=registry,
            max_turns=5,
            max_tool_calls=2,
        ).execute(request)

        self.assertEqual("Ada Lovelace", response.text)
        self.assertEqual(2, response.metadata["tool_calls"])
        self.assertEqual(["p1"], index.read_calls)
        self.assertEqual(
            "qa_read_requires_successful_search",
            response.metadata["react_trace"][0]["public_error_code"],
        )
        self.assertEqual(
            "qa_read_passage_id_not_from_search",
            response.metadata["react_trace"][2]["public_error_code"],
        )

    async def test_react_rejects_initial_completion_then_continues_after_dispatch(self) -> None:
        class EvidenceIndex(FakeIndex):
            def read(self, passage_id: str) -> FakePassage:
                self.read_calls.append(passage_id)
                return FakePassage(
                    passage_id,
                    "d1",
                    "Ada Lovelace",
                    (
                        "Ada Lovelace wrote the first published algorithm. "
                        "Ada Lovelace was an English mathematician."
                    ),
                )

        index = EvidenceIndex()

        def action(name: str, arguments: object) -> str:
            return json.dumps(
                {
                    "arguments": arguments,
                    "kind": "complete" if name == "complete" else "tool",
                    "name": name,
                    "resource_id": (
                        None if name == "complete" else QA_RETRIEVAL_TOOL_ID
                    ),
                    "skill_id": None,
                }
            )

        semantic_artifact = {
            "question_scope": "Who wrote the first published algorithm?",
            "answer_slot": {
                "answer_type": "entity",
                "answer_cardinality": "single",
                "qualifiers": ["first published algorithm"],
                "proposition_index": 0,
                "answer_field": "subject",
            },
            "evidence_propositions": [
                {
                    "subject": "Ada Lovelace",
                    "relation": "wrote",
                    "object_or_attribute_value": "the first published algorithm",
                    "qualifiers": [],
                    "evidence_span": (
                        "Ada Lovelace wrote the first published algorithm."
                    ),
                },
                {
                    "subject": "Ada Lovelace",
                    "relation": "was",
                    "object_or_attribute_value": "English mathematician",
                    "qualifiers": [],
                    "evidence_span": "Ada Lovelace was an English mathematician.",
                },
            ],
            "multi_hop_chain": ["first published algorithm", "Ada Lovelace"],
            "candidate_answer": "Ada Lovelace",
            "evidence": ["Full public text."],
        }

        class SequenceGateway:
            def __init__(self) -> None:
                self.outputs = [
                    action("complete", {"value": "unsupported answer"}),
                    action("search", {"query": "Ada Lovelace", "limit": 1}),
                    action("read", {"passage_id": "p1"}),
                    action("complete", {"value": semantic_artifact}),
                ]
                self.requests: list[AgentRequest] = []

            async def generate(self, request: AgentRequest) -> AgentResponse:
                self.requests.append(request)
                return AgentResponse(self.outputs.pop(0))

        gateway = SequenceGateway()
        request = AgentRequest(
            request_id="qa:completion-admission",
            run_id="qa",
            graph_revision=1,
            problem="Who wrote the first published algorithm?",
            agent=AgentNode(
                "retriever",
                "model",
                "retrieve evidence and answer",
                allowed_tools=(QA_RETRIEVAL_TOOL_ID,),
                execution_mode="react",
                role_family="reasoner",
            ),
            model=ModelSpec("model", "provider"),
            provider=ProviderSpec("provider", kind="test"),
            phase=ExecutionPhase.SINGLE,
            semantic_protocol="hotpotqa_verified_answer_slot_v1",
        )

        response = await QARetrievalReactExecutionAdapter(
            gateway=gateway,
            tool_registry=build_qa_tool_registry(index),
            max_turns=4,
            max_tool_calls=2,
            completion_policy="required_evidence",
        ).execute(request)

        self.assertEqual(
            json.dumps(
                semantic_artifact,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ),
            response.text,
        )
        self.assertEqual(2, response.metadata["tool_calls"])
        self.assertEqual(
            "qa_completion_requires_successful_read_evidence",
            response.metadata["react_trace"][0]["public_error_code"],
        )
        self.assertIn(
            "qa_completion_requires_successful_read_evidence",
            gateway.requests[1].agent.contract,
        )
        self.assertIn(
            "next action must be qa-retrieval search",
            gateway.requests[1].agent.contract,
        )
        self.assertIn(
            "Search arguments contain exactly query and limit",
            gateway.requests[1].agent.contract,
        )
        self.assertIn(
            "next action must be qa-retrieval read",
            gateway.requests[2].agent.contract,
        )
        self.assertIn('"p1"', gateway.requests[2].agent.contract)
        self.assertIn(
            '"arguments":{"passage_id":"p1"}',
            gateway.requests[2].agent.contract,
        )
        self.assertIn(
            "next action must complete",
            gateway.requests[3].agent.contract,
        )
        self.assertIn("Do not use kind=completion", gateway.requests[3].agent.contract)
        search_domain = gateway.requests[1].agent.contract.split(
            "Currently admissible Tool action contracts", 1
        )[1].split("\nCompletion", 1)[0]
        read_domain = gateway.requests[2].agent.contract.split(
            "Currently admissible Tool action contracts", 1
        )[1].split("\nCompletion", 1)[0]
        completion_domain = gateway.requests[3].agent.contract.split(
            "Currently admissible Tool action contracts", 1
        )[1].split("\nCurrently admissible completion", 1)[0]
        self.assertIn('name is "search"', search_domain)
        self.assertNotIn('name is "read"', search_domain)
        self.assertIn("Completion is not admissible", gateway.requests[1].agent.contract)
        self.assertIn('name is "read"', read_domain)
        self.assertNotIn('name is "search"', read_domain)
        self.assertIn("Arguments JSON Schema", search_domain)
        self.assertIn("Arguments JSON Schema", read_domain)
        self.assertNotIn('"argument_json_schema"', search_domain)
        self.assertNotIn('"argument_json_schema"', read_domain)
        self.assertNotIn('"action_envelope"', search_domain)
        self.assertNotIn('"action_envelope"', read_domain)
        self.assertTrue(completion_domain.endswith("- none"))
        self.assertIn(
            "Currently admissible completion schema",
            gateway.requests[3].agent.contract,
        )
        completion_response_schema = json.loads(
            gateway.requests[3].model.metadata["response_json_schema"]
        )
        self.assertEqual(
            ["value"],
            completion_response_schema["properties"]["arguments"]["required"],
        )
        self.assertEqual(
            {
                "question_scope",
                "answer_slot",
                "evidence_propositions",
                "multi_hop_chain",
                "candidate_answer",
                "evidence",
            },
            set(
                completion_response_schema["properties"]["arguments"]
                ["properties"]["value"]["required"]
            ),
        )
        self.assertEqual([("Ada Lovelace", 1)], index.search_calls)
        self.assertEqual(["p1"], index.read_calls)

    async def test_required_evidence_action_domain_rejects_search_after_search(
        self,
    ) -> None:
        index = FakeIndex()

        def action(name: str, arguments: object) -> str:
            return json.dumps(
                {
                    "arguments": arguments,
                    "kind": "complete" if name == "complete" else "tool",
                    "name": name,
                    "resource_id": (
                        None if name == "complete" else QA_RETRIEVAL_TOOL_ID
                    ),
                    "skill_id": None,
                }
            )

        class SequenceGateway:
            def __init__(self) -> None:
                self.outputs = [
                    action("search", {"query": "Ada Lovelace", "limit": 1}),
                    action("search", {"query": "Ada Lovelace", "limit": 1}),
                    action("read", {"passage_id": "p1"}),
                    action("complete", {"value": "Ada Lovelace"}),
                ]

            async def generate(self, request: AgentRequest) -> AgentResponse:
                del request
                return AgentResponse(self.outputs.pop(0))

        request = AgentRequest(
            request_id="qa:state-conditioned-domain",
            run_id="qa",
            graph_revision=1,
            problem="Who wrote the first published algorithm?",
            agent=AgentNode(
                "retriever",
                "model",
                "retrieve evidence and answer",
                allowed_tools=(QA_RETRIEVAL_TOOL_ID,),
                execution_mode="react",
            ),
            model=ModelSpec("model", "provider"),
            provider=ProviderSpec("provider", kind="test"),
            phase=ExecutionPhase.SINGLE,
        )

        response = await QARetrievalReactExecutionAdapter(
            gateway=SequenceGateway(),
            tool_registry=build_qa_tool_registry(index),
            max_turns=4,
            max_tool_calls=2,
            completion_policy="required_evidence",
        ).execute(request)

        self.assertEqual("Ada Lovelace", response.text)
        self.assertEqual(
            "qa_required_evidence_next_action_read",
            response.metadata["react_trace"][1]["public_error_code"],
        )
        self.assertEqual([("Ada Lovelace", 1)], index.search_calls)
        self.assertEqual(["p1"], index.read_calls)

    async def test_hotpot_multi_hop_reads_two_distinct_search_results_before_complete(
        self,
    ) -> None:
        class TwoPassageIndex(FakeIndex):
            def search(self, query: str, *, limit: int) -> tuple[FakeHit, ...]:
                self.search_calls.append((query, limit))
                return (
                    FakeHit("p1", "d1", "Ada Lovelace", "Ada was a mathematician.", 1),
                    FakeHit("p2", "d2", "Algorithm", "The algorithm was published.", 2),
                    FakeHit("p3", "d3", "Program", "The earliest program.", 3),
                )

            def read(self, passage_id: str) -> FakePassage:
                self.read_calls.append(passage_id)
                text = {
                    "p1": "Ada Lovelace was an English mathematician.",
                    "p2": "Ada Lovelace published the first algorithm.",
                    "p3": "Ada Lovelace authored the earliest computer program.",
                }[passage_id]
                return FakePassage(passage_id, passage_id, passage_id, text)

        def action(name: str, arguments: object) -> str:
            return json.dumps(
                {
                    "arguments": arguments,
                    "kind": "complete" if name == "complete" else "tool",
                    "name": name,
                    "resource_id": (
                        None if name == "complete" else QA_RETRIEVAL_TOOL_ID
                    ),
                    "skill_id": None,
                }
            )

        semantic_artifact = {
            "question_scope": "Who published the first algorithm?",
            "answer_slot": {
                "answer_type": "entity",
                "answer_cardinality": "single",
                "qualifiers": ["first algorithm"],
                "proposition_index": 0,
                "answer_field": "subject",
            },
            "evidence_propositions": [
                {
                    "subject": "Ada Lovelace",
                    "relation": "published",
                    "object_or_attribute_value": "the first algorithm",
                    "qualifiers": [],
                    "evidence_span": "Ada Lovelace published the first algorithm.",
                },
                {
                    "subject": "Ada Lovelace",
                    "relation": "was",
                    "object_or_attribute_value": "English mathematician",
                    "qualifiers": [],
                    "evidence_span": "Ada Lovelace was an English mathematician.",
                },
            ],
            "multi_hop_chain": ["Ada Lovelace", "published", "first algorithm"],
            "candidate_answer": "Ada Lovelace",
            "evidence": [
                "Ada Lovelace was an English mathematician.",
                "Ada Lovelace published the first algorithm.",
            ],
        }
        unsupported_span_artifact = json.loads(json.dumps(semantic_artifact))
        unsupported_span_artifact["evidence_propositions"][0][
            "evidence_span"
        ] = "Ada Lovelace authored the earliest computer program."

        class SequenceGateway:
            def __init__(self) -> None:
                self.outputs = [
                    action("search", {"query": "Ada Lovelace algorithm", "limit": 10}),
                    action("read", {"passage_id": "p1"}),
                    action("search", {"query": "first published algorithm", "limit": 10}),
                    action("read", {"passage_id": "p2"}),
                    action(
                        "complete",
                        {"value": unsupported_span_artifact},
                    ),
                    action("search", {"query": "earliest computer program", "limit": 10}),
                    action("read", {"passage_id": "p3"}),
                    action("complete", {"value": semantic_artifact}),
                ]
                self.requests: list[AgentRequest] = []

            async def generate(self, request: AgentRequest) -> AgentResponse:
                self.requests.append(request)
                return AgentResponse(self.outputs.pop(0))

        index = TwoPassageIndex()
        gateway = SequenceGateway()
        request = AgentRequest(
            request_id="qa:two-hop",
            run_id="qa",
            graph_revision=1,
            problem="Who published the first algorithm?",
            agent=AgentNode(
                "reasoner",
                "model",
                "retrieve both supporting passages and determine the answer",
                role_family="reasoner",
                allowed_tools=(QA_RETRIEVAL_TOOL_ID,),
                execution_mode="react",
            ),
            model=ModelSpec("model", "provider"),
            provider=ProviderSpec("provider", kind="test"),
            phase=ExecutionPhase.SINGLE,
            semantic_protocol="hotpotqa_verified_answer_slot_v1",
        )

        response = await QARetrievalReactExecutionAdapter(
            gateway=gateway,
            tool_registry=build_qa_tool_registry(index),
            max_turns=8,
            max_tool_calls=6,
            task_type="multi_hop_qa",
            completion_policy="required_evidence",
        ).execute(request)

        self.assertEqual(json.dumps(semantic_artifact, ensure_ascii=False, sort_keys=True, separators=(",", ":")), response.text)
        self.assertEqual(6, response.metadata["tool_calls"])
        self.assertEqual(["p1", "p2", "p3"], index.read_calls)
        self.assertEqual(3, len(index.search_calls))
        self.assertTrue(
            response.metadata["react_trace"][4]["public_error_code"].startswith(
                "hotpotqa_semantic_evidence_provenance_invalid:"
            )
        )
        self.assertIn(
            "has no typography-canonical lexical match",
            response.metadata["react_trace"][4]["public_error_code"],
        )
        search_schema = json.loads(
            gateway.requests[0].model.metadata["response_json_schema"]
        )
        self.assertEqual(
            10,
            search_schema["properties"]["arguments"]["properties"]["limit"]["const"],
        )
        second_read_schema = json.loads(
            gateway.requests[3].model.metadata["response_json_schema"]
        )
        self.assertIn(
            "p2",
            second_read_schema["properties"]["arguments"]["properties"]
            ["passage_id"]["enum"],
        )
        self.assertIn("next action must be qa-retrieval search", gateway.requests[2].agent.contract)
        self.assertIn("next action must be qa-retrieval read", gateway.requests[3].agent.contract)
        self.assertIn("next action must complete", gateway.requests[4].agent.contract)
        self.assertIn("next action must be qa-retrieval search", gateway.requests[5].agent.contract)
        self.assertIn(
            "use the admitted qa-retrieval search/read continuation",
            gateway.requests[5].agent.contract,
        )
        self.assertIn("next action must be qa-retrieval read", gateway.requests[6].agent.contract)
        completion_schema = json.loads(
            gateway.requests[4].model.metadata["response_json_schema"]
        )
        semantic_properties = completion_schema["properties"]["arguments"][
            "properties"
        ]["value"]["properties"]
        self.assertEqual(
            "entity",
            semantic_properties["answer_slot"]["properties"]["answer_type"][
                "const"
            ],
        )
        self.assertEqual(
            0,
            semantic_properties["answer_slot"]["properties"][
                "proposition_index"
            ]["minimum"],
        )
        self.assertEqual(
            "single",
            semantic_properties["answer_slot"]["properties"][
                "answer_cardinality"
            ]["const"],
        )
        self.assertEqual(
            {
                "answer_type",
                "answer_cardinality",
                "qualifiers",
                "proposition_index",
                "answer_field",
            },
            set(semantic_properties["answer_slot"]["required"]),
        )

    async def test_hotpot_reasoner_accepts_direct_retriever_read_provenance_without_spending_own_tool_budget(
        self,
    ) -> None:
        evidence_text = (
            "Ada Lovelace was an English mathematician. "
            "Ada Lovelace published the first algorithm."
        )
        upstream_read_receipt = {
            "tool_id": QA_RETRIEVAL_TOOL_ID,
            "tool_version": "frozen-index-v1",
            "request": {
                "action": "read",
                "arguments": {"passage_id": "p1"},
            },
            "result": {
                "value": {
                    "operation": "read",
                    "passage_id": "p1",
                    "passage": {
                        "passage_id": "p1",
                        "text": evidence_text,
                    },
                },
                "completed": True,
            },
            "error_type": None,
        }
        artifact = {
            "question_scope": "Who published the first algorithm?",
            "answer_slot": {
                "answer_type": "entity",
                "answer_cardinality": "single",
                "qualifiers": ["first algorithm"],
                "proposition_index": 0,
                "answer_field": "subject",
            },
            "evidence_propositions": [
                {
                    "subject": "Ada Lovelace",
                    "relation": "published",
                    "object_or_attribute_value": "the first algorithm",
                    "qualifiers": [],
                    "evidence_span": (
                        "Ada Lovelace published the first algorithm."
                    ),
                },
                {
                    "subject": "Ada Lovelace",
                    "relation": "was",
                    "object_or_attribute_value": "English mathematician",
                    "qualifiers": [],
                    "evidence_span": (
                        "Ada Lovelace was an English mathematician."
                    ),
                },
            ],
            "multi_hop_chain": [
                "Ada Lovelace --occupation--> English mathematician",
                "Ada Lovelace --published--> the first algorithm",
            ],
            "candidate_answer": "Ada Lovelace",
            "evidence": [evidence_text],
        }

        class CompletionGateway:
            async def generate(self, request: AgentRequest) -> AgentResponse:
                del request
                return AgentResponse(
                    json.dumps(
                        {
                            "arguments": {"value": artifact},
                            "kind": "complete",
                            "name": "complete",
                            "resource_id": None,
                            "skill_id": None,
                        }
                    )
                )

        request = AgentRequest(
            request_id="qa:upstream-retrieval-provenance",
            run_id="qa",
            graph_revision=1,
            problem="Who published the first algorithm?",
            agent=AgentNode(
                "reasoner",
                "model",
                "bind direct predecessor evidence to the answer slot",
                role_family="reasoner",
                allowed_tools=(QA_RETRIEVAL_TOOL_ID,),
                execution_mode="react",
            ),
            model=ModelSpec("model", "provider"),
            provider=ProviderSpec("provider", kind="test"),
            phase=ExecutionPhase.SINGLE,
            semantic_protocol="hotpotqa_verified_answer_slot_v1",
            upstream=(
                UpstreamMessage(
                    "evidence_retriever",
                    "reasoner",
                    "retrieved passage p1",
                    graph_revision=1,
                    artifact_type="retrieval_evidence",
                    tool_receipts=(upstream_read_receipt,),
                ),
            ),
        )

        response = await QARetrievalReactExecutionAdapter(
            gateway=CompletionGateway(),
            tool_registry=build_qa_tool_registry(FakeIndex()),
            max_turns=1,
            max_tool_calls=2,
            task_type="multi_hop_qa",
            completion_policy="required_evidence",
        ).execute(request)

        self.assertEqual(
            json.dumps(
                artifact,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ),
            response.text,
        )
        self.assertEqual(0, response.metadata["tool_calls"])
        self.assertEqual(0, response.metadata["continued_tool_receipt_count"])
        self.assertEqual([], response.metadata["tool_receipts"])

    async def test_hotpot_reasoner_rejects_forged_span_despite_upstream_read_receipt(
        self,
    ) -> None:
        upstream_read_receipt = {
            "tool_id": QA_RETRIEVAL_TOOL_ID,
            "tool_version": "frozen-index-v1",
            "request": {
                "action": "read",
                "arguments": {"passage_id": "p1"},
            },
            "result": {
                "value": {
                    "operation": "read",
                    "passage_id": "p1",
                    "passage": {
                        "passage_id": "p1",
                        "text": (
                            "Ada Lovelace was an English mathematician. "
                            "Ada Lovelace published the first algorithm."
                        ),
                    },
                },
                "completed": True,
            },
            "error_type": None,
        }
        artifact = {
            "question_scope": "Who published the first algorithm?",
            "answer_slot": {
                "answer_type": "entity",
                "answer_cardinality": "single",
                "qualifiers": ["first algorithm"],
                "proposition_index": 0,
                "answer_field": "subject",
            },
            "evidence_propositions": [
                {
                    "subject": "Ada Lovelace",
                    "relation": "published",
                    "object_or_attribute_value": "the first algorithm",
                    "qualifiers": [],
                    "evidence_span": "A forged passage states the answer.",
                },
                {
                    "subject": "Ada Lovelace",
                    "relation": "was",
                    "object_or_attribute_value": "English mathematician",
                    "qualifiers": [],
                    "evidence_span": (
                        "Ada Lovelace was an English mathematician."
                    ),
                },
            ],
            "multi_hop_chain": [
                "Ada Lovelace --occupation--> English mathematician",
                "Ada Lovelace --published--> the first algorithm",
            ],
            "candidate_answer": "Ada Lovelace",
            "evidence": ["A forged passage states the answer."],
        }

        class CompletionGateway:
            async def generate(self, request: AgentRequest) -> AgentResponse:
                del request
                return AgentResponse(
                    json.dumps(
                        {
                            "arguments": {"value": artifact},
                            "kind": "complete",
                            "name": "complete",
                            "resource_id": None,
                            "skill_id": None,
                        }
                    )
                )

        request = AgentRequest(
            request_id="qa:forged-upstream-retrieval-provenance",
            run_id="qa",
            graph_revision=1,
            problem="Who published the first algorithm?",
            agent=AgentNode(
                "reasoner",
                "model",
                "bind direct predecessor evidence to the answer slot",
                role_family="reasoner",
                allowed_tools=(QA_RETRIEVAL_TOOL_ID,),
                execution_mode="react",
            ),
            model=ModelSpec("model", "provider"),
            provider=ProviderSpec("provider", kind="test"),
            phase=ExecutionPhase.SINGLE,
            semantic_protocol="hotpotqa_verified_answer_slot_v1",
            upstream=(
                UpstreamMessage(
                    "evidence_retriever",
                    "reasoner",
                    "retrieved passage p1",
                    graph_revision=1,
                    artifact_type="retrieval_evidence",
                    tool_receipts=(upstream_read_receipt,),
                ),
            ),
        )

        with self.assertRaises(ReactExecutionError) as caught:
            await QARetrievalReactExecutionAdapter(
                gateway=CompletionGateway(),
                tool_registry=build_qa_tool_registry(FakeIndex()),
                max_turns=1,
                max_tool_calls=2,
                task_type="multi_hop_qa",
                completion_policy="required_evidence",
            ).execute(request)

        self.assertTrue(
            caught.exception.react_trace[-1]["public_error_code"].startswith(
                "hotpotqa_semantic_evidence_provenance_invalid:"
            )
        )
        self.assertEqual(0, len(caught.exception.tool_receipts))

    async def test_hotpot_provenance_rejection_marks_tool_plan_exhausted_when_pair_cannot_fit(
        self,
    ) -> None:
        upstream_read_receipt = {
            "tool_id": QA_RETRIEVAL_TOOL_ID,
            "tool_version": "frozen-index-v1",
            "request": {
                "action": "read",
                "arguments": {"passage_id": "p1"},
            },
            "result": {
                "value": {
                    "operation": "read",
                    "passage_id": "p1",
                    "passage": {
                        "passage_id": "p1",
                        "text": (
                            "Ada Lovelace was an English mathematician. "
                            "Ada Lovelace published the first algorithm."
                        ),
                    },
                },
                "completed": True,
            },
            "error_type": None,
        }
        artifact = {
            "question_scope": "Who published the first algorithm?",
            "answer_slot": {
                "answer_type": "entity",
                "answer_cardinality": "single",
                "qualifiers": ["first algorithm"],
                "proposition_index": 0,
                "answer_field": "subject",
            },
            "evidence_propositions": [
                {
                    "subject": "Ada Lovelace",
                    "relation": "published",
                    "object_or_attribute_value": "the first algorithm",
                    "qualifiers": [],
                    "evidence_span": "A forged passage states the answer.",
                },
                {
                    "subject": "Ada Lovelace",
                    "relation": "was",
                    "object_or_attribute_value": "English mathematician",
                    "qualifiers": [],
                    "evidence_span": (
                        "Ada Lovelace was an English mathematician."
                    ),
                },
            ],
            "multi_hop_chain": [
                "Ada Lovelace --occupation--> English mathematician",
                "Ada Lovelace --published--> the first algorithm",
            ],
            "candidate_answer": "Ada Lovelace",
            "evidence": ["A forged passage states the answer."],
        }

        class CompletionGateway:
            async def generate(self, request: AgentRequest) -> AgentResponse:
                del request
                return AgentResponse(
                    json.dumps(
                        {
                            "arguments": {"value": artifact},
                            "kind": "complete",
                            "name": "complete",
                            "resource_id": None,
                            "skill_id": None,
                        }
                    )
                )

        request = AgentRequest(
            request_id="qa:upstream-provenance-tool-plan-exhausted",
            run_id="qa",
            graph_revision=1,
            problem="Who published the first algorithm?",
            agent=AgentNode(
                "reasoner",
                "model",
                "bind direct predecessor evidence to the answer slot",
                role_family="reasoner",
                allowed_tools=(QA_RETRIEVAL_TOOL_ID,),
                execution_mode="react",
            ),
            model=ModelSpec("model", "provider"),
            provider=ProviderSpec("provider", kind="test"),
            phase=ExecutionPhase.SINGLE,
            semantic_protocol="hotpotqa_verified_answer_slot_v1",
            upstream=(
                UpstreamMessage(
                    "evidence_retriever",
                    "reasoner",
                    "retrieved passage p1",
                    graph_revision=1,
                    artifact_type="retrieval_evidence",
                    tool_receipts=(upstream_read_receipt,),
                ),
            ),
        )

        with self.assertRaises(ReactExecutionError) as caught:
            await QARetrievalReactExecutionAdapter(
                gateway=CompletionGateway(),
                tool_registry=build_qa_tool_registry(FakeIndex()),
                max_turns=1,
                max_tool_calls=1,
                task_type="multi_hop_qa",
                completion_policy="required_evidence",
            ).execute(request)

        self.assertIs(True, caught.exception.tool_plan_exhausted)
        terminal_diagnosis = caught.exception.react_trace[-1][
            "terminal_failure_diagnosis"
        ]
        self.assertIs(True, terminal_diagnosis["tool_plan_exhausted"])
        self.assertEqual(1, terminal_diagnosis["remaining_tool_calls"])
        self.assertEqual(0, len(caught.exception.tool_receipts))

    async def test_structural_semantic_rejection_repairs_with_existing_evidence(
        self,
    ) -> None:
        class TwoPassageIndex(FakeIndex):
            def search(self, query: str, *, limit: int) -> tuple[FakeHit, ...]:
                self.search_calls.append((query, limit))
                return (
                    FakeHit("p1", "d1", "Ada Lovelace", "Occupation.", 1),
                    FakeHit("p2", "d2", "Algorithm", "Publication.", 2),
                )

            def read(self, passage_id: str) -> FakePassage:
                self.read_calls.append(passage_id)
                text = {
                    "p1": "Ada Lovelace was an English mathematician.",
                    "p2": "Ada Lovelace published the first algorithm.",
                }[passage_id]
                return FakePassage(passage_id, passage_id, passage_id, text)

        def action(name: str, arguments: object) -> str:
            return json.dumps(
                {
                    "arguments": arguments,
                    "kind": "complete" if name == "complete" else "tool",
                    "name": name,
                    "resource_id": (
                        None if name == "complete" else QA_RETRIEVAL_TOOL_ID
                    ),
                    "skill_id": None,
                }
            )

        valid_artifact = {
            "question_scope": "Who published the first algorithm?",
            "answer_slot": {
                "answer_type": "entity",
                "answer_cardinality": "single",
                "qualifiers": ["first algorithm"],
                "proposition_index": 0,
                "answer_field": "subject",
            },
            "evidence_propositions": [
                {
                    "subject": "Ada Lovelace",
                    "relation": "published",
                    "object_or_attribute_value": "the first algorithm",
                    "qualifiers": [],
                    "evidence_span": "Ada Lovelace published the first algorithm.",
                },
                {
                    "subject": "Ada Lovelace",
                    "relation": "was",
                    "object_or_attribute_value": "English mathematician",
                    "qualifiers": [],
                    "evidence_span": "Ada Lovelace was an English mathematician.",
                },
            ],
            "multi_hop_chain": ["Ada Lovelace", "published", "first algorithm"],
            "candidate_answer": "Ada Lovelace",
            "evidence": [
                "Ada Lovelace was an English mathematician.",
                "Ada Lovelace published the first algorithm.",
            ],
        }
        binding_mismatch = json.loads(json.dumps(valid_artifact))
        binding_mismatch["candidate_answer"] = "the first algorithm"

        class SequenceGateway:
            def __init__(self) -> None:
                self.outputs = [
                    action("search", {"query": "Ada Lovelace", "limit": 10}),
                    action("read", {"passage_id": "p1"}),
                    action("search", {"query": "first algorithm", "limit": 10}),
                    action("read", {"passage_id": "p2"}),
                    action("complete", {"value": binding_mismatch}),
                    action("complete", {"value": valid_artifact}),
                ]
                self.requests: list[AgentRequest] = []

            async def generate(self, request: AgentRequest) -> AgentResponse:
                self.requests.append(request)
                return AgentResponse(self.outputs.pop(0))

        index = TwoPassageIndex()
        gateway = SequenceGateway()
        request = AgentRequest(
            request_id="qa:structural-semantic-repair",
            run_id="qa",
            graph_revision=1,
            problem="Who published the first algorithm?",
            agent=AgentNode(
                "reasoner",
                "model",
                "retrieve both supporting passages and determine the answer",
                role_family="reasoner",
                allowed_tools=(QA_RETRIEVAL_TOOL_ID,),
                execution_mode="react",
            ),
            model=ModelSpec("model", "provider"),
            provider=ProviderSpec("provider", kind="test"),
            phase=ExecutionPhase.SINGLE,
            semantic_protocol="hotpotqa_verified_answer_slot_v1",
        )

        response = await QARetrievalReactExecutionAdapter(
            gateway=gateway,
            tool_registry=build_qa_tool_registry(index),
            max_turns=6,
            max_tool_calls=4,
            task_type="multi_hop_qa",
            completion_policy="required_evidence",
        ).execute(request)

        self.assertEqual(
            json.dumps(
                valid_artifact,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ),
            response.text,
        )
        self.assertEqual(4, response.metadata["tool_calls"])
        self.assertEqual(2, len(index.search_calls))
        self.assertEqual(["p1", "p2"], index.read_calls)
        structural_error = response.metadata["react_trace"][4][
            "public_error_code"
        ]
        self.assertTrue(
            structural_error.startswith("hotpotqa_semantic_artifact_invalid:")
        )
        self.assertIn("answer_slot.answer_field selects", structural_error)
        self.assertIn(
            "object_or_attribute_value",
            structural_error,
        )

        repair_contract = gateway.requests[5].agent.contract
        self.assertIn(structural_error, repair_contract)
        self.assertIn("set answer_slot.proposition_index", repair_contract)
        self.assertIn(
            "answer_slot.answer_field to that same proposition field",
            repair_contract,
        )
        self.assertIn("Repair only those answer_slot fields", repair_contract)
        self.assertIn(
            "minimal but complete evidence-aligned referential surface",
            repair_contract,
        )
        self.assertIn("complete possessor entity mention", repair_contract)
        self.assertIn("possessive marker", repair_contract)
        self.assertIn("Do not search or read again", repair_contract)
        repair_domain = repair_contract.split(
            "Currently admissible Tool action contracts", 1
        )[1].split("\nCurrently admissible completion", 1)[0]
        self.assertTrue(repair_domain.endswith("- none"))
        self.assertIn("Currently admissible completion schema", repair_contract)

    async def test_unified_entity_binding_repairs_on_preserved_read_evidence(
        self,
    ) -> None:
        class DenchIndex(FakeIndex):
            def search(self, query: str, *, limit: int) -> tuple[FakeHit, ...]:
                self.search_calls.append((query, limit))
                return (
                    FakeHit(
                        "p1",
                        "d1",
                        "Judi Dench",
                        "Dench was born in Heworth.",
                        1,
                    ),
                )

            def read(self, passage_id: str) -> FakePassage:
                self.read_calls.append(passage_id)
                return FakePassage(
                    passage_id,
                    "d1",
                    "Judi Dench",
                    "Dench was born in Heworth, North Riding of Yorkshire.",
                )

        # This regression isolates same-read entity binding.  Explicit named-
        # scope location resolution is covered independently above.
        question = "Where was Dame Judi Dench born?"
        valid_artifact = {
            "question_scope": question,
            "answer_slot": {
                "answer_type": "location",
                "answer_cardinality": "single",
                "qualifiers": ["in England"],
                "proposition_index": 0,
                "answer_field": "object_or_attribute_value",
            },
            "evidence_propositions": [
                {
                    "subject": "Dench",
                    "relation": "was born in",
                    "object_or_attribute_value": (
                        "Heworth, North Riding of Yorkshire"
                    ),
                    "qualifiers": ["in England"],
                    "evidence_span": (
                        "Dench was born in Heworth, North Riding of Yorkshire."
                    ),
                }
            ],
            "multi_hop_chain": [
                "Dench --was born in--> Heworth, North Riding of Yorkshire"
            ],
            "candidate_answer": "Heworth, North Riding of Yorkshire",
            "evidence": ["p1"],
        }
        ungrounded_entity_surface = json.loads(json.dumps(valid_artifact))
        ungrounded_entity_surface["evidence_propositions"][0]["subject"] = (
            "Dame Judi Dench"
        )

        def action(name: str, arguments: object) -> str:
            return json.dumps(
                {
                    "arguments": arguments,
                    "kind": "complete" if name == "complete" else "tool",
                    "name": name,
                    "resource_id": (
                        None if name == "complete" else QA_RETRIEVAL_TOOL_ID
                    ),
                    "skill_id": None,
                }
            )

        class SequenceGateway:
            def __init__(self) -> None:
                self.outputs = [
                    action(
                        "search",
                        {"query": "Dame Judi Dench birthplace", "limit": 5},
                    ),
                    action("read", {"passage_id": "p1"}),
                    action("complete", {"value": ungrounded_entity_surface}),
                    action("complete", {"value": valid_artifact}),
                ]
                self.requests: list[AgentRequest] = []

            async def generate(self, request: AgentRequest) -> AgentResponse:
                self.requests.append(request)
                return AgentResponse(self.outputs.pop(0))

        index = DenchIndex()
        gateway = SequenceGateway()
        request = AgentRequest(
            request_id="trivia:entity-binding-repair",
            run_id="trivia",
            graph_revision=1,
            problem=question,
            agent=AgentNode(
                "reasoner",
                "model",
                "retrieve evidence and preserve the requested relation",
                role_family="reasoner",
                allowed_tools=(QA_RETRIEVAL_TOOL_ID,),
                execution_mode="react",
            ),
            model=ModelSpec("model", "provider"),
            provider=ProviderSpec("provider", kind="test"),
            phase=ExecutionPhase.SINGLE,
            semantic_protocol=QA_VERIFIED_ANSWER_LINEAGE_PROTOCOL,
        )

        response = await QARetrievalReactExecutionAdapter(
            gateway=gateway,
            tool_registry=build_qa_tool_registry(index),
            max_turns=4,
            max_tool_calls=10,
            task_type="factual_qa",
            completion_policy="required_evidence",
        ).execute(request)

        self.assertEqual([("Dame Judi Dench birthplace", 5)], index.search_calls)
        self.assertEqual(["p1"], index.read_calls)
        public_error = response.metadata["react_trace"][2]["public_error_code"]
        self.assertTrue(public_error.startswith("qa_semantic_artifact_invalid:"))
        self.assertIn("subject is not grounded", public_error)
        repair_contract = gateway.requests[3].agent.contract
        self.assertIn("Do not search or read again", repair_contract)
        self.assertIn("Repair only the named proposition argument", repair_contract)

    async def test_react_failed_retrieval_receipt_admits_explicit_completion(self) -> None:
        class FailingIndex(FakeIndex):
            def search(self, query: str, *, limit: int) -> tuple[FakeHit, ...]:
                self.search_calls.append((query, limit))
                raise TimeoutError("public retrieval timeout")

        def action(name: str, arguments: object) -> str:
            return json.dumps(
                {
                    "arguments": arguments,
                    "kind": "complete" if name == "complete" else "tool",
                    "name": name,
                    "resource_id": (
                        None if name == "complete" else QA_RETRIEVAL_TOOL_ID
                    ),
                    "skill_id": None,
                }
            )

        class SequenceGateway:
            def __init__(self) -> None:
                self.outputs = [
                    action("search", {"query": "Ada Lovelace", "limit": 1}),
                    action("complete", {"value": "insufficient evidence"}),
                ]

            async def generate(self, request: AgentRequest) -> AgentResponse:
                del request
                return AgentResponse(self.outputs.pop(0))

        response = await QARetrievalReactExecutionAdapter(
            gateway=SequenceGateway(),
            tool_registry=build_qa_tool_registry(FailingIndex()),
            max_turns=2,
            max_tool_calls=1,
        ).execute(
            AgentRequest(
                request_id="qa:failed-retrieval",
                run_id="qa",
                graph_revision=1,
                problem="Who wrote the first published algorithm?",
                agent=AgentNode(
                    "retriever",
                    "model",
                    "retrieve evidence and answer",
                    allowed_tools=(QA_RETRIEVAL_TOOL_ID,),
                    execution_mode="react",
                ),
                model=ModelSpec("model", "provider"),
                provider=ProviderSpec("provider", kind="test"),
                phase=ExecutionPhase.SINGLE,
            )
        )

        self.assertEqual("insufficient evidence", response.text)
        self.assertEqual(1, response.metadata["tool_calls"])
        self.assertEqual("TimeoutError", response.metadata["tool_receipts"][0]["error_type"])

    async def test_required_evidence_rejects_failed_retrieval_receipt(self) -> None:
        class FailingIndex(FakeIndex):
            def search(self, query: str, *, limit: int) -> tuple[FakeHit, ...]:
                self.search_calls.append((query, limit))
                raise TimeoutError("public retrieval timeout")

        def action(name: str, arguments: object) -> str:
            return json.dumps(
                {
                    "arguments": arguments,
                    "kind": "complete" if name == "complete" else "tool",
                    "name": name,
                    "resource_id": (
                        None if name == "complete" else QA_RETRIEVAL_TOOL_ID
                    ),
                    "skill_id": None,
                }
            )

        class SequenceGateway:
            def __init__(self) -> None:
                self.outputs = [
                    action("search", {"query": "Ada Lovelace", "limit": 1}),
                    action("complete", {"value": "unsupported answer"}),
                ]

            async def generate(self, request: AgentRequest) -> AgentResponse:
                del request
                return AgentResponse(self.outputs.pop(0))

        with self.assertRaises(ReactExecutionError) as caught:
            await QARetrievalReactExecutionAdapter(
                gateway=SequenceGateway(),
                tool_registry=build_qa_tool_registry(FailingIndex()),
                max_turns=2,
                max_tool_calls=1,
                completion_policy="required_evidence",
            ).execute(
                AgentRequest(
                    request_id="qa:required-evidence-failure",
                    run_id="qa",
                    graph_revision=1,
                    problem="Who wrote the first published algorithm?",
                    agent=AgentNode(
                        "retriever",
                        "model",
                        "retrieve evidence and answer",
                        allowed_tools=(QA_RETRIEVAL_TOOL_ID,),
                        execution_mode="react",
                    ),
                    model=ModelSpec("model", "provider"),
                    provider=ProviderSpec("provider", kind="test"),
                    phase=ExecutionPhase.SINGLE,
                )
            )

        error = caught.exception
        self.assertEqual("TimeoutError", error.tool_receipts[0]["error_type"])
        self.assertEqual(
            "qa_completion_requires_successful_read_evidence",
            error.react_trace[-1]["public_error_code"],
        )

    def test_evidence_retriever_schema_and_contract_are_answer_free(self) -> None:
        adapter = QARetrievalReactExecutionAdapter(
            gateway=SimpleNamespace(generate=lambda request: None),
            tool_registry=build_qa_tool_registry(FakeIndex()),
            max_turns=3,
            max_tool_calls=2,
            task_type="factual_qa",
            completion_policy="required_evidence",
        )
        request = AgentRequest(
            request_id="trivia:evidence-retriever-schema",
            run_id="trivia",
            graph_revision=1,
            problem="Which city does David Soul come from?",
            agent=AgentNode(
                "evidence_retriever",
                "model",
                "retrieve evidence for the requested entity and relation",
                role_family="evidence_retriever",
                allowed_tools=(QA_RETRIEVAL_TOOL_ID,),
                execution_mode="react",
            ),
            model=ModelSpec("model", "provider"),
            provider=ProviderSpec("provider", kind="test"),
            phase=ExecutionPhase.SINGLE,
            semantic_protocol=QA_VERIFIED_ANSWER_LINEAGE_PROTOCOL,
        )

        schema = adapter._completion_arguments_schema(request)
        artifact_schema = schema["properties"]["value"]
        expected_fields = {
            "question_scope",
            "entity_identity",
            "target_relation",
            "answer_type_constraint",
            "evidence_proposition",
            "evidence_span",
            "passage_id",
        }
        self.assertEqual(expected_fields, set(artifact_schema["required"]))
        self.assertEqual(expected_fields, set(artifact_schema["properties"]))
        self.assertFalse(artifact_schema["additionalProperties"])
        self.assertEqual(
            {"question_surface", "evidence_surface"},
            set(
                artifact_schema["properties"]["entity_identity"]["required"]
            ),
        )
        for answer_field in ("candidate_answer", "answer_slot", "final_answer"):
            self.assertNotIn(answer_field, artifact_schema["properties"])
        self.assertIn(
            "question-side entity or event anchor",
            artifact_schema["properties"]["entity_identity"]["properties"][
                "question_surface"
            ]["description"],
        )
        self.assertIn(
            "must not be a wh-word or wh-phrase",
            artifact_schema["properties"]["entity_identity"]["properties"][
                "question_surface"
            ]["description"],
        )
        self.assertIn(
            "coreferential surface of the same",
            artifact_schema["properties"]["entity_identity"]["properties"][
                "evidence_surface"
            ]["description"],
        )
        self.assertIn(
            "question-side requested relation",
            artifact_schema["properties"]["target_relation"]["description"],
        )
        self.assertIn(
            "exact receipt-grounded predicate",
            artifact_schema["properties"]["target_relation"]["description"],
        )
        self.assertEqual(
            "entity",
            artifact_schema["properties"]["answer_type_constraint"]["const"],
        )
        proposition_schema = artifact_schema["properties"]["evidence_proposition"]
        self.assertEqual(
            {"subject", "predicate", "object_or_attribute_value"},
            set(proposition_schema["required"]),
        )

        completion_contract = adapter._contract(
            request,
            [
                {
                    "observation_status": "success",
                    "result": {
                        "operation": "read",
                        "passage_id": "p1",
                        "passage": {
                            "passage_id": "p1",
                            "text": "David Soul is from Chicago.",
                        },
                    },
                }
            ],
        )
        self.assertIn("As the Evidence Retriever", completion_contract)
        self.assertIn(
            "Cite one successful qa-retrieval read receipt",
            completion_contract,
        )
        self.assertIn(
            "Do not select or emit candidate_answer, answer_slot, or final_answer",
            completion_contract,
        )
        self.assertIn("question-side entity/event anchor", completion_contract)
        self.assertIn("never a wh-word/wh-phrase", completion_contract)
        self.assertIn(
            "passage title for explicit coreference binding",
            completion_contract,
        )
        self.assertIn(
            "question_surface and evidence_surface differ",
            completion_contract,
        )
        self.assertIn(
            "evidence_surface may be a coreferential pronoun or short-name surface",
            completion_contract,
        )
        self.assertIn(
            "may occupy the subject, the object, or neither",
            completion_contract,
        )

    def test_evidence_retriever_accepts_david_soul_title_alias_binding(
        self,
    ) -> None:
        question = "Which city does David Soul come from?"
        evidence = "Soul comes from Chicago, Illinois."
        artifact = {
            "question_scope": question,
            "entity_identity": {
                "question_surface": "David Soul",
                "evidence_surface": "Soul",
            },
            "target_relation": "comes from",
            "answer_type_constraint": "entity",
            "evidence_proposition": {
                "subject": "Soul",
                "predicate": "comes from",
                "object_or_attribute_value": "Chicago, Illinois",
            },
            "evidence_span": evidence,
            "passage_id": "david-soul",
        }
        receipt = {
            "tool_id": QA_RETRIEVAL_TOOL_ID,
            "tool_version": "frozen-index-v1",
            "request": {
                "action": "read",
                "arguments": {"passage_id": "david-soul"},
            },
            "result": {
                "value": {
                    "operation": "read",
                    "passage_id": "david-soul",
                    "passage": {
                        "passage_id": "david-soul",
                        "title": "David Soul",
                        "text": evidence,
                    },
                },
                "completed": True,
            },
            "error_type": None,
        }

        issue = QARetrievalReactExecutionAdapter._evidence_retriever_completion_issue
        self.assertIsNone(
            issue(
                original_question=question,
                artifact=json.dumps(artifact),
                tool_receipts=[receipt],
            )
        )

        wh_anchor = json.loads(json.dumps(artifact))
        wh_anchor["entity_identity"]["question_surface"] = "Which city"
        detail = issue(
            original_question=question,
            artifact=json.dumps(wh_anchor),
            tool_receipts=[receipt],
        )
        self.assertIsNotNone(detail)
        assert detail is not None
        self.assertIn("not a wh-word or wh-phrase", detail)

    def test_evidence_retriever_preserves_complete_answer_bearing_title_identity(
        self,
    ) -> None:
        question = (
            "Who received the Royal Medal in 1848, Ada Lovelace or "
            "Charles Babbage?"
        )

        def completion_issue(
            *,
            evidence: str,
            evidence_surface: str,
            subject: str,
            title: str = "Ada Lovelace (mathematician)",
        ) -> str | None:
            artifact = {
                "question_scope": question,
                "entity_identity": {
                    "question_surface": "Ada Lovelace",
                    "evidence_surface": evidence_surface,
                },
                "target_relation": "received the Royal Medal in 1848",
                "answer_type_constraint": "entity",
                "evidence_proposition": {
                    "subject": subject,
                    "predicate": "received",
                    "object_or_attribute_value": "the Royal Medal",
                },
                "evidence_span": evidence,
                "passage_id": "ada-lovelace",
            }
            receipt = {
                "tool_id": QA_RETRIEVAL_TOOL_ID,
                "tool_version": "frozen-index-v1",
                "request": {
                    "action": "read",
                    "arguments": {"passage_id": "ada-lovelace"},
                },
                "result": {
                    "value": {
                        "operation": "read",
                        "passage_id": "ada-lovelace",
                        "passage": {
                            "passage_id": "ada-lovelace",
                            "title": title,
                            "text": evidence,
                        },
                    },
                    "completed": True,
                },
                "error_type": None,
            }
            return QARetrievalReactExecutionAdapter._evidence_retriever_completion_issue(
                original_question=question,
                artifact=json.dumps(artifact),
                tool_receipts=[receipt],
            )

        short_surface_issue = completion_issue(
            evidence="In 1848, Lovelace received the Royal Medal.",
            evidence_surface="Lovelace",
            subject="Lovelace",
        )
        self.assertIsNotNone(short_surface_issue)
        assert short_surface_issue is not None
        self.assertIn("strict subset", short_surface_issue)
        self.assertIn(
            "complete receipt-grounded entity mention",
            short_surface_issue,
        )

        self.assertIsNone(
            completion_issue(
                evidence="In 1848, Ada Lovelace received the Royal Medal.",
                evidence_surface="Ada Lovelace",
                subject="Ada Lovelace",
            )
        )
        self.assertIsNone(
            completion_issue(
                evidence=(
                    "Ada Lovelace Augusta Ada Lovelace (10 December 1815 "
                    "– 27 November 1852) received the Royal Medal in 1848."
                ),
                evidence_surface="Ada Lovelace",
                subject="Augusta Ada Lovelace",
                title="Ada Lovelace",
            )
        )

        expanded_identity_issue = completion_issue(
            evidence=(
                "Ada Lovelace Augusta Ada Lovelace (10 December 1815 "
                "– 27 November 1852) received the Royal Medal in 1848."
            ),
            evidence_surface="Augusta Ada Lovelace",
            subject="Augusta Ada Lovelace",
            title="Ada Lovelace",
        )
        self.assertIsNotNone(expanded_identity_issue)
        assert expanded_identity_issue is not None
        self.assertIn(
            "passage title identity chain",
            expanded_identity_issue,
        )

        descriptive_surface_issue = completion_issue(
            evidence=(
                "In 1848, American writer Ada Lovelace received the Royal Medal."
            ),
            evidence_surface="American writer Ada Lovelace",
            subject="American writer Ada Lovelace",
            title="Ada Lovelace",
        )
        self.assertIsNotNone(descriptive_surface_issue)
        assert descriptive_surface_issue is not None
        self.assertIn("passage title identity chain", descriptive_surface_issue)

        title_case_descriptor_issue = completion_issue(
            evidence=(
                "Ada Lovelace American Writer Ada Lovelace (10 December "
                "1815 – 27 November 1852) received the Royal Medal in 1848."
            ),
            evidence_surface="American Writer Ada Lovelace",
            subject="American Writer Ada Lovelace",
            title="Ada Lovelace",
        )
        self.assertIsNotNone(title_case_descriptor_issue)
        assert title_case_descriptor_issue is not None
        self.assertIn(
            "passage title identity chain",
            title_case_descriptor_issue,
        )

    def test_evidence_retriever_accepts_receipt_grounded_inflection_and_rejects_relation_drift(
        self,
    ) -> None:
        question = (
            "In which decade did Billboard magazine first publish an American "
            "hit chart?"
        )
        evidence = (
            "On January 4, 1936, Billboard magazine published its first music "
            "hit parade."
        )
        artifact = {
            "question_scope": question,
            "entity_identity": {
                "question_surface": "Billboard magazine",
                "evidence_surface": "Billboard magazine",
            },
            "target_relation": "publish",
            "answer_type_constraint": "date",
            "evidence_proposition": {
                "subject": "Billboard magazine",
                "predicate": "published",
                "object_or_attribute_value": "1936",
            },
            "evidence_span": evidence,
            "passage_id": "billboard-history",
        }
        receipt = {
            "tool_id": QA_RETRIEVAL_TOOL_ID,
            "tool_version": "frozen-index-v1",
            "request": {
                "action": "read",
                "arguments": {"passage_id": "billboard-history"},
            },
            "result": {
                "value": {
                    "operation": "read",
                    "passage_id": "billboard-history",
                    "passage": {
                        "passage_id": "billboard-history",
                        "title": "Billboard magazine",
                        "text": evidence,
                    },
                },
                "completed": True,
            },
            "error_type": None,
        }
        issue = QARetrievalReactExecutionAdapter._evidence_retriever_completion_issue
        self.assertIsNone(
            issue(
                original_question=question,
                artifact=json.dumps(artifact),
                tool_receipts=[receipt],
            )
        )
        wrong_relation = json.loads(json.dumps(artifact))
        wrong_relation["target_relation"] = "introduced"
        detail = issue(
            original_question=question,
            artifact=json.dumps(wrong_relation),
            tool_receipts=[receipt],
        )
        self.assertIsNotNone(detail)
        assert detail is not None
        self.assertIn("target_relation must preserve", detail)

    def test_evidence_retriever_separates_question_relation_from_receipt_predicate(
        self,
    ) -> None:
        question = (
            "Who won the Nobel Prize in Literature in 1930, Sinclair Lewis "
            "or Ernest Hemingway?"
        )
        evidence = (
            "Sinclair Lewis was an American novelist. In 1930, he became the "
            "first writer from the United States to receive the Nobel Prize "
            "in Literature."
        )
        artifact = {
            "question_scope": question,
            "entity_identity": {
                "question_surface": "Sinclair Lewis",
                "evidence_surface": "he",
            },
            "target_relation": "won the Nobel Prize in Literature in 1930",
            "answer_type_constraint": "entity",
            "evidence_proposition": {
                "subject": "he",
                "predicate": "became",
                "object_or_attribute_value": (
                    "the first writer from the United States to receive the "
                    "Nobel Prize in Literature"
                ),
            },
            "evidence_span": evidence,
            "passage_id": "sinclair-lewis",
        }
        receipt = {
            "tool_id": QA_RETRIEVAL_TOOL_ID,
            "tool_version": "frozen-index-v1",
            "request": {
                "action": "read",
                "arguments": {"passage_id": "sinclair-lewis"},
            },
            "result": {
                "value": {
                    "operation": "read",
                    "passage_id": "sinclair-lewis",
                    "passage": {
                        "passage_id": "sinclair-lewis",
                        "title": "Sinclair Lewis",
                        "text": evidence,
                    },
                },
                "completed": True,
            },
            "error_type": None,
        }
        issue = QARetrievalReactExecutionAdapter._evidence_retriever_completion_issue
        self.assertIsNone(
            issue(
                original_question=question,
                artifact=json.dumps(artifact),
                tool_receipts=[receipt],
            )
        )

        unbound_identity = json.loads(json.dumps(artifact))
        unbound_identity["entity_identity"]["evidence_surface"] = (
            "Sinclair Lewis"
        )
        detail = issue(
            original_question=question,
            artifact=json.dumps(unbound_identity),
            tool_receipts=[receipt],
        )
        self.assertIsNotNone(detail)
        assert detail is not None
        self.assertIn("exactly one evidence_proposition relation argument", detail)

        unrelated_relation = json.loads(json.dumps(artifact))
        unrelated_relation["evidence_proposition"]["predicate"] = "was"
        unrelated_relation["evidence_proposition"][
            "object_or_attribute_value"
        ] = "an American novelist"
        detail = issue(
            original_question=question,
            artifact=json.dumps(unrelated_relation),
            tool_receipts=[receipt],
        )
        self.assertIsNotNone(detail)
        assert detail is not None
        self.assertIn("lack controlled relation alignment", detail)

        title_only = json.loads(json.dumps(artifact))
        title_only["entity_identity"]["evidence_surface"] = "Sinclair"
        title_only["evidence_span"] = "Sinclair Lewis"
        title_only["evidence_proposition"] = {
            "subject": "Sinclair",
            "predicate": "Lewis",
            "object_or_attribute_value": "Sinclair Lewis",
        }
        detail = issue(
            original_question=question,
            artifact=json.dumps(title_only),
            tool_receipts=[receipt],
        )
        self.assertIsNotNone(detail)
        assert detail is not None
        self.assertIn("not the cited passage title alone", detail)

    def test_evidence_retriever_binds_ordinal_to_relation_scope_in_one_proposition(
        self,
    ) -> None:
        question = "In which year did the Society first publish an index?"

        def completion_issue(
            *,
            evidence: str,
            relation: str,
            passage_id: str,
        ) -> str | None:
            artifact = {
                "question_scope": question,
                "entity_identity": {
                    "question_surface": "the Society",
                    "evidence_surface": "The Society",
                },
                "target_relation": relation,
                "answer_type_constraint": "date",
                "evidence_proposition": {
                    "subject": "The Society",
                    "predicate": relation,
                    "object_or_attribute_value": "1990",
                },
                "evidence_span": evidence,
                "passage_id": passage_id,
            }
            receipt = {
                "tool_id": QA_RETRIEVAL_TOOL_ID,
                "tool_version": "frozen-index-v1",
                "request": {
                    "action": "read",
                    "arguments": {"passage_id": passage_id},
                },
                "result": {
                    "value": {
                        "operation": "read",
                        "passage_id": passage_id,
                        "passage": {
                            "passage_id": passage_id,
                            "title": "The Society",
                            "text": evidence,
                        },
                    },
                    "completed": True,
                },
                "error_type": None,
            }
            return QARetrievalReactExecutionAdapter._evidence_retriever_completion_issue(
                original_question=question,
                artifact=json.dumps(artifact),
                tool_receipts=[receipt],
            )

        unrelated_ordinal = completion_issue(
            evidence=(
                "The Society received its first award in 1988. "
                "The Society published the index in 1990."
            ),
            relation="published",
            passage_id="unrelated-ordinal",
        )
        self.assertIsNotNone(unrelated_ordinal)
        assert unrelated_ordinal is not None
        self.assertIn("qa_ordinal_relation_scope_mismatch", unrelated_ordinal)
        self.assertIn("same receipt-grounded proposition", unrelated_ordinal)

        local_subtype = completion_issue(
            evidence="The Society first published a regional index in 1990.",
            relation="published",
            passage_id="local-subtype",
        )
        self.assertIsNotNone(local_subtype)
        assert local_subtype is not None
        self.assertIn("qa_ordinal_relation_scope_mismatch", local_subtype)
        self.assertIn("local/subtype scope", local_subtype)

        self.assertIsNone(
            completion_issue(
                evidence="The Society first published the index in 1990.",
                relation="published",
                passage_id="same-proposition",
            )
        )

        # ``began publishing`` is a relation/onset paraphrase of the question's
        # ``first publish`` scope.  It remains answer-free and must still bind
        # entity, relation, ordinal onset, and date in the same proposition.
        self.assertIsNone(
            completion_issue(
                evidence="The Society began publishing the index in 1990.",
                relation="began publishing",
                passage_id="semantic-relation-surface",
            )
        )

    def test_reasoner_rejects_ordinal_from_unrelated_fact_in_same_read(
        self,
    ) -> None:
        question = "In which decade did the Society first publish an index?"
        evidence = (
            "The Society published the index in 1990 after receiving its first "
            "award."
        )
        artifact = {
            "question_scope": question,
            "answer_slot": {
                "answer_type": "date",
                "answer_cardinality": "single",
                "qualifiers": ["first"],
                "proposition_index": 0,
                "answer_field": "object_or_attribute_value",
            },
            "evidence_propositions": [
                {
                    "subject": "The Society",
                    "relation": "published",
                    "object_or_attribute_value": "1990",
                    "qualifiers": ["first"],
                    "evidence_span": evidence,
                }
            ],
            "multi_hop_chain": ["The Society --published--> 1990"],
            "candidate_answer": "1990s",
            "evidence": ["society-history"],
        }
        receipt = {
            "tool_id": QA_RETRIEVAL_TOOL_ID,
            "tool_version": "frozen-index-v1",
            "request": {
                "action": "read",
                "arguments": {"passage_id": "society-history"},
            },
            "result": {
                "value": {
                    "operation": "read",
                    "passage_id": "society-history",
                    "passage": {
                        "passage_id": "society-history",
                        "title": "The Society",
                        "text": evidence,
                    },
                },
                "completed": True,
            },
            "error_type": None,
        }
        adapter = QARetrievalReactExecutionAdapter(
            gateway=SimpleNamespace(generate=lambda request: None),
            tool_registry=build_qa_tool_registry(FakeIndex()),
            max_turns=4,
            max_tool_calls=10,
            task_type="factual_qa",
            completion_policy="required_evidence",
        )
        question_token = adapter._semantic_reasoner_question.set(question)
        protocol_token = adapter._semantic_reasoner_protocol.set(
            QA_VERIFIED_ANSWER_LINEAGE_PROTOCOL
        )
        retrieval_token = adapter._retrieval_completion_required.set(True)
        try:
            detail = adapter._completion_error(
                action=StructuredAction(
                    kind=ActionKind.COMPLETE,
                    name="complete",
                    arguments={"value": artifact},
                ),
                artifact=json.dumps(artifact),
                tool_receipts=[receipt],
            )
        finally:
            adapter._retrieval_completion_required.reset(retrieval_token)
            adapter._semantic_reasoner_protocol.reset(protocol_token)
            adapter._semantic_reasoner_question.reset(question_token)
        self.assertIsNotNone(detail)
        assert detail is not None
        self.assertTrue(detail.startswith("qa_semantic_evidence_provenance_invalid:"))
        self.assertIn("qa_ordinal_relation_scope_mismatch", detail)
        self.assertIn("same receipt-grounded proposition", detail)

    def test_factual_evidence_retriever_honorific_title_alias_chain_is_exact(
        self,
    ) -> None:
        question = "Where in England was Dame Judi Dench born?"
        evidence = "Dench was born in Heworth, North Riding of Yorkshire."
        artifact = {
            "question_scope": question,
            "entity_identity": {
                "question_surface": "Dame Judi Dench",
                "evidence_surface": "Dench",
            },
            "target_relation": "was born in",
            "answer_type_constraint": "location",
            "evidence_proposition": {
                "subject": "Dench",
                "predicate": "was born in",
                "object_or_attribute_value": (
                    "Heworth, North Riding of Yorkshire"
                ),
            },
            "evidence_span": evidence,
            "passage_id": "dench-birthplace",
        }

        def receipt(title: str | None) -> dict[str, object]:
            passage = {
                "passage_id": "dench-birthplace",
                "text": evidence,
            }
            if title is not None:
                passage["title"] = title
            return {
                "tool_id": QA_RETRIEVAL_TOOL_ID,
                "tool_version": "frozen-index-v1",
                "request": {
                    "action": "read",
                    "arguments": {"passage_id": "dench-birthplace"},
                },
                "result": {
                    "value": {
                        "operation": "read",
                        "passage_id": "dench-birthplace",
                        "passage": passage,
                    },
                    "completed": True,
                },
                "error_type": None,
            }

        issue = QARetrievalReactExecutionAdapter._evidence_retriever_completion_issue
        self.assertIsNone(
            issue(
                original_question=question,
                artifact=json.dumps(artifact),
                tool_receipts=[receipt("Judi Dench")],
            )
        )

        for title in (None, "Judy Dench"):
            with self.subTest(title=title):
                detail = issue(
                    original_question=question,
                    artifact=json.dumps(artifact),
                    tool_receipts=[receipt(title)],
                )
                self.assertIsNotNone(detail)
                assert detail is not None
                self.assertIn("alias identity lacks an explicit binding", detail)

        answer_in_identity = json.loads(json.dumps(artifact))
        answer_in_identity["entity_identity"]["evidence_surface"] = (
            "Heworth, North Riding of Yorkshire"
        )
        detail = issue(
            original_question=question,
            artifact=json.dumps(answer_in_identity),
            tool_receipts=[receipt("Judi Dench")],
        )
        self.assertIsNotNone(detail)
        assert detail is not None
        self.assertIn("passage title identity chain", detail)

    def test_title_bound_identity_may_bind_reverse_relation_object(self) -> None:
        question = "Who was represented by Dr Alice Carter?"
        evidence = "Morgan Reed was represented by Carter."
        artifact = {
            "question_scope": question,
            "entity_identity": {
                "question_surface": "Dr Alice Carter",
                "evidence_surface": "Carter",
            },
            "target_relation": "was represented by",
            "answer_type_constraint": "entity",
            "evidence_proposition": {
                "subject": "Morgan Reed",
                "predicate": "was represented by",
                "object_or_attribute_value": "Carter",
            },
            "evidence_span": evidence,
            "passage_id": "alice-carter",
        }
        receipt = {
            "tool_id": QA_RETRIEVAL_TOOL_ID,
            "tool_version": "frozen-index-v1",
            "request": {
                "action": "read",
                "arguments": {"passage_id": "alice-carter"},
            },
            "result": {
                "value": {
                    "operation": "read",
                    "passage_id": "alice-carter",
                    "passage": {
                        "passage_id": "alice-carter",
                        "title": "Alice Carter",
                        "text": evidence,
                    },
                },
                "completed": True,
            },
            "error_type": None,
        }

        self.assertIsNone(
            QARetrievalReactExecutionAdapter._evidence_retriever_completion_issue(
                original_question=question,
                artifact=json.dumps(artifact),
                tool_receipts=[receipt],
            )
        )

    async def test_factual_identity_repair_stays_completion_only_at_tool_budget(
        self,
    ) -> None:
        question = "Where in England was Dame Judi Dench born?"
        evidence = "Dench was born in Heworth, North Riding of Yorkshire."
        valid_artifact = {
            "question_scope": question,
            "entity_identity": {
                "question_surface": "Dame Judi Dench",
                "evidence_surface": "Dench",
            },
            "target_relation": "was born in",
            "answer_type_constraint": "location",
            "evidence_proposition": {
                "subject": "Dench",
                "predicate": "was born in",
                "object_or_attribute_value": (
                    "Heworth, North Riding of Yorkshire"
                ),
            },
            "evidence_span": evidence,
            "passage_id": "p1",
        }
        invalid_artifact = json.loads(json.dumps(valid_artifact))
        invalid_artifact["entity_identity"]["evidence_surface"] = (
            "Heworth, North Riding of Yorkshire"
        )

        def action(name: str, arguments: object) -> str:
            return json.dumps(
                {
                    "arguments": arguments,
                    "kind": "complete" if name == "complete" else "tool",
                    "name": name,
                    "resource_id": (
                        None if name == "complete" else QA_RETRIEVAL_TOOL_ID
                    ),
                    "skill_id": None,
                }
            )

        class DenchIndex(FakeIndex):
            def read(self, passage_id: str) -> FakePassage:
                self.read_calls.append(passage_id)
                return FakePassage(passage_id, "d1", "Judi Dench", evidence)

        class SequenceGateway:
            def __init__(self) -> None:
                self.requests: list[AgentRequest] = []
                self.outputs = [
                    action(
                        "search",
                        {"query": "Dame Judi Dench birthplace", "limit": 5},
                    ),
                    action("read", {"passage_id": "p1"}),
                    action("complete", {"value": invalid_artifact}),
                    action("complete", {"value": valid_artifact}),
                ]

            async def generate(self, request: AgentRequest) -> AgentResponse:
                self.requests.append(request)
                return AgentResponse(self.outputs.pop(0))

        index = DenchIndex()
        gateway = SequenceGateway()
        response = await QARetrievalReactExecutionAdapter(
            gateway=gateway,
            tool_registry=build_qa_tool_registry(index),
            max_turns=4,
            max_tool_calls=2,
            task_type="factual_qa",
            completion_policy="required_evidence",
        ).execute(
            AgentRequest(
                request_id="trivia:honorific-identity-repair",
                run_id="trivia",
                graph_revision=1,
                problem=question,
                agent=AgentNode(
                    "evidence_retriever",
                    "model",
                    "retrieve receipt-grounded identity and birthplace evidence",
                    role_family="evidence_retriever",
                    allowed_tools=(QA_RETRIEVAL_TOOL_ID,),
                    execution_mode="react",
                ),
                model=ModelSpec("model", "provider"),
                provider=ProviderSpec("provider", kind="test"),
                phase=ExecutionPhase.SINGLE,
                semantic_protocol=QA_VERIFIED_ANSWER_LINEAGE_PROTOCOL,
            )
        )

        self.assertEqual([("Dame Judi Dench birthplace", 5)], index.search_calls)
        self.assertEqual(["p1"], index.read_calls)
        self.assertEqual(2, response.metadata["tool_calls"])
        feedback = response.metadata["react_trace"][2]["public_error_code"]
        self.assertTrue(feedback.startswith("qa_semantic_artifact_invalid:"))
        self.assertNotIn("knowledge_base_coverage_failure", feedback)
        repair_contract = gateway.requests[3].agent.contract
        self.assertIn("Emit a complete action; do not search or read again", repair_contract)
        self.assertIn(
            "- none\nCurrently admissible completion schema",
            repair_contract,
        )

        exhausted_gateway = SequenceGateway()
        exhausted_gateway.outputs[-1] = action(
            "complete",
            {"value": invalid_artifact},
        )
        with self.assertRaises(ReactExecutionError) as exhausted:
            await QARetrievalReactExecutionAdapter(
                gateway=exhausted_gateway,
                tool_registry=build_qa_tool_registry(DenchIndex()),
                max_turns=4,
                max_tool_calls=2,
                task_type="factual_qa",
                completion_policy="required_evidence",
            ).execute(
                AgentRequest(
                    request_id="trivia:honorific-identity-repair-exhausted",
                    run_id="trivia",
                    graph_revision=1,
                    problem=question,
                    agent=AgentNode(
                        "evidence_retriever",
                        "model",
                        "retrieve receipt-grounded identity and birthplace evidence",
                        role_family="evidence_retriever",
                        allowed_tools=(QA_RETRIEVAL_TOOL_ID,),
                        execution_mode="react",
                    ),
                    model=ModelSpec("model", "provider"),
                    provider=ProviderSpec("provider", kind="test"),
                    phase=ExecutionPhase.SINGLE,
                    semantic_protocol=QA_VERIFIED_ANSWER_LINEAGE_PROTOCOL,
                )
            )
        exact_repair = (
            QARetrievalReactExecutionAdapter._public_semantic_repair_instruction(
                exhausted.exception.react_trace[-1]["public_error_code"]
            )
        )
        self.assertIsNotNone(exact_repair)
        self.assertEqual(
            exact_repair,
            exhausted.exception.react_trace[-1]["repair_instruction"],
        )
        assert exact_repair is not None
        self.assertIn(
            exact_repair,
            exhausted_gateway.requests[-1].agent.contract,
        )

    def test_evidence_retriever_allows_open_super_bowl_event_anchor(
        self,
    ) -> None:
        question = "Who won Super Bowl XX?"
        evidence = (
            "Super Bowl XX matched the Chicago Bears and New England Patriots. "
            "The Bears defeated the Patriots by 46-10."
        )
        artifact = {
            "question_scope": question,
            "entity_identity": {
                "question_surface": "Super Bowl XX",
                "evidence_surface": "Super Bowl XX",
            },
            "target_relation": "defeated",
            "answer_type_constraint": "entity",
            "evidence_proposition": {
                "subject": "The Bears",
                "predicate": "defeated",
                "object_or_attribute_value": "the Patriots",
            },
            "evidence_span": evidence,
            "passage_id": "super-bowl-xx",
        }
        receipt = {
            "tool_id": QA_RETRIEVAL_TOOL_ID,
            "tool_version": "frozen-index-v1",
            "request": {
                "action": "read",
                "arguments": {"passage_id": "super-bowl-xx"},
            },
            "result": {
                "value": {
                    "operation": "read",
                    "passage_id": "super-bowl-xx",
                    "passage": {
                        "passage_id": "super-bowl-xx",
                        "title": "Super Bowl XX",
                        "text": evidence,
                    },
                },
                "completed": True,
            },
            "error_type": None,
        }

        issue = QARetrievalReactExecutionAdapter._evidence_retriever_completion_issue
        self.assertIsNone(
            issue(
                original_question=question,
                artifact=json.dumps(artifact),
                tool_receipts=[receipt],
            )
        )
        for answer_field in ("candidate_answer", "answer_slot", "final_answer"):
            self.assertNotIn(answer_field, artifact)

        both_arguments = json.loads(json.dumps(artifact))
        both_arguments["target_relation"] = "won"
        both_arguments["evidence_proposition"] = {
            "subject": "Super Bowl XX",
            "predicate": "won",
            "object_or_attribute_value": "Super Bowl XX",
        }
        both_arguments["evidence_span"] = "Super Bowl XX won Super Bowl XX."
        both_receipt = json.loads(json.dumps(receipt))
        both_receipt["result"]["value"]["passage"]["text"] = both_arguments[
            "evidence_span"
        ]
        detail = issue(
            original_question=question,
            artifact=json.dumps(both_arguments),
            tool_receipts=[both_receipt],
        )
        self.assertIsNotNone(detail)
        assert detail is not None
        self.assertIn("must not bind", detail)
        self.assertIn("both relation arguments", detail)

    async def test_evidence_retriever_completes_one_grounded_read_artifact(
        self,
    ) -> None:
        question = "Which city does David Soul come from?"
        evidence = "David Soul is from Chicago."
        artifact = {
            "question_scope": question,
            "entity_identity": {
                "question_surface": "David Soul",
                "evidence_surface": "David Soul",
            },
            "target_relation": "is from",
            "answer_type_constraint": "entity",
            "evidence_proposition": {
                "subject": "David Soul",
                "predicate": "is from",
                "object_or_attribute_value": "Chicago",
            },
            "evidence_span": evidence,
            "passage_id": "p1",
        }

        def action(name: str, arguments: object) -> str:
            return json.dumps(
                {
                    "arguments": arguments,
                    "kind": "complete" if name == "complete" else "tool",
                    "name": name,
                    "resource_id": (
                        None if name == "complete" else QA_RETRIEVAL_TOOL_ID
                    ),
                    "skill_id": None,
                }
            )

        class SoulIndex(FakeIndex):
            def read(self, passage_id: str) -> FakePassage:
                self.read_calls.append(passage_id)
                return FakePassage(passage_id, "d1", "David Soul", evidence)

        class SequenceGateway:
            def __init__(self) -> None:
                self.outputs = [
                    action("search", {"query": "David Soul origin city", "limit": 5}),
                    action("read", {"passage_id": "p1"}),
                    action("complete", {"value": artifact}),
                ]

            async def generate(self, request: AgentRequest) -> AgentResponse:
                del request
                return AgentResponse(self.outputs.pop(0))

        index = SoulIndex()
        response = await QARetrievalReactExecutionAdapter(
            gateway=SequenceGateway(),
            tool_registry=build_qa_tool_registry(index),
            max_turns=3,
            max_tool_calls=2,
            task_type="factual_qa",
            completion_policy="required_evidence",
        ).execute(
            AgentRequest(
                request_id="trivia:evidence-retriever-grounded",
                run_id="trivia",
                graph_revision=1,
                problem=question,
                agent=AgentNode(
                    "evidence_retriever",
                    "model",
                    "retrieve receipt-grounded entity and relation evidence",
                    role_family="evidence_retriever",
                    allowed_tools=(QA_RETRIEVAL_TOOL_ID,),
                    execution_mode="react",
                ),
                model=ModelSpec("model", "provider"),
                provider=ProviderSpec("provider", kind="test"),
                phase=ExecutionPhase.SINGLE,
                semantic_protocol=QA_VERIFIED_ANSWER_LINEAGE_PROTOCOL,
            )
        )

        self.assertEqual(
            json.dumps(
                artifact,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ),
            response.text,
        )
        self.assertEqual([("David Soul origin city", 5)], index.search_calls)
        self.assertEqual(["p1"], index.read_calls)
        self.assertEqual(2, response.metadata["tool_calls"])

    async def test_evidence_span_lexical_mismatch_repairs_on_preserved_read(
        self,
    ) -> None:
        question = "Who won Super Bowl XX?"
        evidence = (
            "Super Bowl XX matched the Chicago Bears and New England Patriots. "
            "The Bears defeated the Patriots by 46-10."
        )
        valid_artifact = {
            "question_scope": question,
            "entity_identity": {
                "question_surface": "Super Bowl XX",
                "evidence_surface": "Super Bowl XX",
            },
            "target_relation": "defeated",
            "answer_type_constraint": "entity",
            "evidence_proposition": {
                "subject": "The Bears",
                "predicate": "defeated",
                "object_or_attribute_value": "the Patriots",
            },
            "evidence_span": evidence,
            "passage_id": "p1",
        }
        invalid_artifact = {
            **valid_artifact,
            "evidence_span": (
                "Super Bowl XX was won when the Bears beat the Patriots 46-10."
            ),
        }

        def action(name: str, arguments: object) -> str:
            return json.dumps(
                {
                    "arguments": arguments,
                    "kind": "complete" if name == "complete" else "tool",
                    "name": name,
                    "resource_id": (
                        None if name == "complete" else QA_RETRIEVAL_TOOL_ID
                    ),
                    "skill_id": None,
                }
            )

        class SuperBowlIndex(FakeIndex):
            def read(self, passage_id: str) -> FakePassage:
                self.read_calls.append(passage_id)
                return FakePassage(passage_id, "d1", "Super Bowl XX", evidence)

        class SequenceGateway:
            def __init__(self) -> None:
                self.requests: list[AgentRequest] = []
                self.outputs = [
                    action("search", {"query": "Super Bowl XX winner", "limit": 5}),
                    action("read", {"passage_id": "p1"}),
                    action("complete", {"value": invalid_artifact}),
                    action("complete", {"value": valid_artifact}),
                ]

            async def generate(self, request: AgentRequest) -> AgentResponse:
                self.requests.append(request)
                return AgentResponse(self.outputs.pop(0))

        index = SuperBowlIndex()
        gateway = SequenceGateway()
        response = await QARetrievalReactExecutionAdapter(
            gateway=gateway,
            tool_registry=build_qa_tool_registry(index),
            max_turns=4,
            max_tool_calls=10,
            task_type="factual_qa",
            completion_policy="required_evidence",
        ).execute(
            AgentRequest(
                request_id="trivia:super-bowl-span-repair",
                run_id="trivia",
                graph_revision=1,
                problem=question,
                agent=AgentNode(
                    "evidence_retriever",
                    "model",
                    "retrieve receipt-grounded entity and relation evidence",
                    role_family="evidence_retriever",
                    allowed_tools=(QA_RETRIEVAL_TOOL_ID,),
                    execution_mode="react",
                ),
                model=ModelSpec("model", "provider"),
                provider=ProviderSpec("provider", kind="test"),
                phase=ExecutionPhase.SINGLE,
                semantic_protocol=QA_VERIFIED_ANSWER_LINEAGE_PROTOCOL,
            )
        )

        self.assertEqual([("Super Bowl XX winner", 5)], index.search_calls)
        self.assertEqual(["p1"], index.read_calls)
        self.assertEqual(2, response.metadata["tool_calls"])
        lexical_feedback = response.metadata["react_trace"][2][
            "public_error_code"
        ]
        self.assertTrue(lexical_feedback.startswith("qa_semantic_artifact_invalid:"))
        self.assertIn("typography-canonical lexical match", lexical_feedback)
        self.assertNotIn("knowledge_base_coverage_failure", lexical_feedback)
        repair_contract = gateway.requests[3].agent.contract
        self.assertIn("Preserve the cited passage_id", repair_contract)
        self.assertIn("do not paraphrase", repair_contract)
        self.assertIn(
            "- none\nCurrently admissible completion schema",
            repair_contract,
        )

    async def test_evidence_retriever_relation_field_repairs_on_preserved_read(
        self,
    ) -> None:
        question = "Where in England was Dame Judi Dench born?"
        evidence = "Dench was born in Heworth, North Riding of Yorkshire."
        base_artifact = {
            "question_scope": question,
            "entity_identity": {
                "question_surface": "Dench",
                "evidence_surface": "Dench",
            },
            "answer_type_constraint": "location",
            "evidence_proposition": {
                "subject": "Dench",
                "predicate": "was born in",
                "object_or_attribute_value": "Heworth, North Riding of Yorkshire",
            },
            "evidence_span": evidence,
            "passage_id": "p1",
        }

        def action(name: str, arguments: object) -> str:
            return json.dumps(
                {
                    "arguments": arguments,
                    "kind": "complete" if name == "complete" else "tool",
                    "name": name,
                    "resource_id": (
                        None if name == "complete" else QA_RETRIEVAL_TOOL_ID
                    ),
                    "skill_id": None,
                }
            )

        class DenchIndex(FakeIndex):
            def search(self, query: str, *, limit: int) -> tuple[FakeHit, ...]:
                self.search_calls.append((query, limit))
                passage_id = f"p{len(self.search_calls)}"
                return (
                    FakeHit(
                        passage_id,
                        f"d{len(self.search_calls)}",
                        "Judi Dench",
                        evidence,
                        1,
                    ),
                )

            def read(self, passage_id: str) -> FakePassage:
                self.read_calls.append(passage_id)
                return FakePassage(passage_id, "d1", "Judi Dench", evidence)

        class SequenceGateway:
            def __init__(self) -> None:
                self.requests: list[AgentRequest] = []
                self.outputs = [
                    action(
                        "search",
                        {"query": "Dame Judi Dench birthplace", "limit": 5},
                    ),
                    action("read", {"passage_id": "p1"}),
                    action(
                        "complete",
                        {
                            "value": {
                                **base_artifact,
                                "entity_identity": {
                                    "question_surface": question,
                                    "evidence_surface": evidence,
                                },
                                "target_relation": "birthplace",
                            }
                        },
                    ),
                    action(
                        "complete",
                        {
                            "value": {
                                **base_artifact,
                                "entity_identity": {
                                    "question_surface": "Dench",
                                    "evidence_surface": evidence,
                                },
                                "target_relation": "birthplace",
                            }
                        },
                    ),
                    action(
                        "complete",
                        {"value": {**base_artifact, "target_relation": "birthplace"}},
                    ),
                    action(
                        "complete",
                        {
                            "value": {
                                **base_artifact,
                                "target_relation": "was born in",
                            }
                        },
                    ),
                ]

            async def generate(self, request: AgentRequest) -> AgentResponse:
                self.requests.append(request)
                return AgentResponse(self.outputs.pop(0))

        index = DenchIndex()
        gateway = SequenceGateway()
        response = await QARetrievalReactExecutionAdapter(
            gateway=gateway,
            tool_registry=build_qa_tool_registry(index),
            max_turns=8,
            max_tool_calls=10,
            task_type="factual_qa",
            completion_policy="required_evidence",
        ).execute(
            AgentRequest(
                request_id="trivia:evidence-retriever-relation-repair",
                run_id="trivia",
                graph_revision=1,
                problem=question,
                agent=AgentNode(
                    "evidence_retriever",
                    "model",
                    "retrieve receipt-grounded entity and relation evidence",
                    role_family="evidence_retriever",
                    allowed_tools=(QA_RETRIEVAL_TOOL_ID,),
                    execution_mode="react",
                ),
                model=ModelSpec("model", "provider"),
                provider=ProviderSpec("provider", kind="test"),
                phase=ExecutionPhase.SINGLE,
                semantic_protocol=QA_VERIFIED_ANSWER_LINEAGE_PROTOCOL,
            )
        )

        self.assertEqual(
            [("Dame Judi Dench birthplace", 5)],
            index.search_calls,
        )
        self.assertEqual(["p1"], index.read_calls)
        public_errors = [
            response.metadata["react_trace"][index]["public_error_code"]
            for index in (2, 3, 4)
        ]
        self.assertTrue(
            all(
                error.startswith("qa_semantic_artifact_invalid:")
                for error in public_errors
            )
        )
        self.assertIn("whole original question", public_errors[0])
        self.assertIn("whole evidence_span", public_errors[1])
        self.assertIn("target_relation must preserve", public_errors[2])
        for request in gateway.requests[3:6]:
            self.assertIn("do not add a search or read", request.agent.contract)
        target_relation_repair = gateway.requests[5].agent.contract
        self.assertIn("Repair only target_relation", target_relation_repair)
        self.assertIn("same successful read receipt", target_relation_repair)
        for preserved_field in (
            "passage_id",
            "question_scope",
            "entity_identity",
            "answer_type_constraint",
            "evidence_span",
            "evidence_proposition",
        ):
            with self.subTest(preserved_field=preserved_field):
                self.assertIn(preserved_field, target_relation_repair)
        self.assertIn("Do not search or read again", target_relation_repair)
        self.assertEqual(
            {**base_artifact, "target_relation": "was born in"},
            json.loads(response.text),
        )
        self.assertEqual(2, response.metadata["tool_calls"])

    def test_evidence_retriever_rejects_unprovenanced_or_answer_only_artifacts(
        self,
    ) -> None:
        question = "Which city does David Soul come from?"
        evidence = "David Soul comes from Chicago."
        receipt = {
            "tool_id": QA_RETRIEVAL_TOOL_ID,
            "tool_version": "frozen-index-v1",
            "request": {
                "action": "read",
                "arguments": {"passage_id": "p1"},
            },
            "result": {
                "value": {
                    "operation": "read",
                    "passage_id": "p1",
                    "passage": {"passage_id": "p1", "text": evidence},
                },
                "completed": True,
            },
            "error_type": None,
        }
        valid = {
            "question_scope": question,
            "entity_identity": {
                "question_surface": "David Soul",
                "evidence_surface": "David Soul",
            },
            "target_relation": "comes from",
            "answer_type_constraint": "entity",
            "evidence_proposition": {
                "subject": "David Soul",
                "predicate": "comes from",
                "object_or_attribute_value": "Chicago",
            },
            "evidence_span": evidence,
            "passage_id": "p1",
        }

        issue = QARetrievalReactExecutionAdapter._evidence_retriever_completion_issue
        self.assertIsNone(
            issue(
                original_question=question,
                artifact=json.dumps(valid),
                tool_receipts=[receipt],
            )
        )
        whitespace_only = json.loads(json.dumps(valid))
        whitespace_only["target_relation"] = " comes from "
        whitespace_only["evidence_span"] = evidence + " "
        self.assertIsNone(
            issue(
                original_question=question,
                artifact=json.dumps(whitespace_only),
                tool_receipts=[receipt],
            )
        )

        invalid_cases: dict[str, tuple[object, str]] = {}
        for name, path, value, expected in (
            (
                "passage",
                ("passage_id",),
                "p2",
                "no matching successful",
            ),
            (
                "span",
                ("evidence_span",),
                "David Soul comes from London.",
                "no typography-canonical lexical match",
            ),
            (
                "question entity",
                ("entity_identity", "question_surface"),
                "David Crockett",
                "does not occur in the original question",
            ),
            (
                "evidence entity",
                ("entity_identity", "evidence_surface"),
                "David Crockett",
                "does not occur in evidence_span",
            ),
            (
                "relation",
                ("target_relation",),
                "was born in",
                "target_relation must preserve",
            ),
        ):
            candidate = json.loads(json.dumps(valid))
            target = candidate
            for key in path[:-1]:
                target = target[key]
            target[path[-1]] = value
            invalid_cases[name] = (candidate, expected)
        invalid_cases["bare answer"] = (
            "London",
            "labelled artifact must begin with a declared `Field:` label",
        )

        for name, (candidate, expected) in invalid_cases.items():
            with self.subTest(name=name):
                detail = issue(
                    original_question=question,
                    artifact=(
                        candidate
                        if isinstance(candidate, str)
                        else json.dumps(candidate)
                    ),
                    tool_receipts=[receipt],
                )
                self.assertIsNotNone(detail)
                assert detail is not None
                self.assertIn(expected, detail)

    def test_evidence_retriever_alias_requires_explicit_identity_binding(
        self,
    ) -> None:
        question = "Where was Norma Jeane Mortenson born?"

        def receipt(passage_id: str, text: str) -> dict[str, object]:
            return {
                "tool_id": QA_RETRIEVAL_TOOL_ID,
                "tool_version": "frozen-index-v1",
                "request": {
                    "action": "read",
                    "arguments": {"passage_id": passage_id},
                },
                "result": {
                    "value": {
                        "operation": "read",
                        "passage_id": passage_id,
                        "passage": {
                            "passage_id": passage_id,
                            "text": text,
                        },
                    },
                    "completed": True,
                },
                "error_type": None,
            }

        positive_text = (
            "Norma Jeane Mortenson, known as Marilyn Monroe, was born in "
            "Los Angeles."
        )
        positive = {
            "question_scope": question,
            "entity_identity": {
                "question_surface": "Norma Jeane Mortenson",
                "evidence_surface": "Marilyn Monroe",
            },
            "target_relation": "was born in",
            "answer_type_constraint": "location",
            "evidence_proposition": {
                "subject": "Marilyn Monroe",
                "predicate": "was born in",
                "object_or_attribute_value": "Los Angeles",
            },
            "evidence_span": positive_text,
            "passage_id": "positive",
        }
        issue = QARetrievalReactExecutionAdapter._evidence_retriever_completion_issue
        self.assertIsNone(
            issue(
                original_question=question,
                artifact=json.dumps(positive),
                tool_receipts=[receipt("positive", positive_text)],
            )
        )

        negative_text = "Marilyn Monroe was born in Los Angeles."
        negative = {
            **positive,
            "evidence_span": negative_text,
            "passage_id": "negative",
        }
        detail = issue(
            original_question=question,
            artifact=json.dumps(negative),
            tool_receipts=[receipt("negative", negative_text)],
        )
        self.assertIsNotNone(detail)
        assert detail is not None
        self.assertIn("alias identity lacks an explicit binding", detail)

    def test_evidence_retriever_binds_relation_argument_to_question_answer_type(
        self,
    ) -> None:
        def receipt(passage_id: str, text: str) -> dict[str, object]:
            return {
                "tool_id": QA_RETRIEVAL_TOOL_ID,
                "tool_version": "frozen-index-v1",
                "request": {
                    "action": "read",
                    "arguments": {"passage_id": passage_id},
                },
                "result": {
                    "value": {
                        "operation": "read",
                        "passage_id": passage_id,
                        "passage": {
                            "passage_id": passage_id,
                            "title": "Judi Dench",
                            "text": text,
                        },
                    },
                    "completed": True,
                },
                "error_type": None,
            }

        issue = QARetrievalReactExecutionAdapter._evidence_retriever_completion_issue
        question = "Where in England was Dame Judi Dench born?"
        date_evidence = "Dench was born 9 December 1934."
        date_artifact = {
            "question_scope": question,
            "entity_identity": {
                "question_surface": "Dench",
                "evidence_surface": "Dench",
            },
            "target_relation": "was born",
            "answer_type_constraint": "location",
            "evidence_proposition": {
                "subject": "Dench",
                "predicate": "was born",
                "object_or_attribute_value": "9 December 1934",
            },
            "evidence_span": date_evidence,
            "passage_id": "date",
        }
        detail = issue(
            original_question=question,
            artifact=json.dumps(date_artifact),
            tool_receipts=[receipt("date", date_evidence)],
        )
        self.assertIsNotNone(detail)
        assert detail is not None
        self.assertIn("supplies a date relation argument", detail)
        self.assertIn("requested location relation argument", detail)

        identity_and_type_mismatch = json.loads(json.dumps(date_artifact))
        identity_and_type_mismatch["entity_identity"] = {
            "question_surface": "Dame Judi Dench",
            "evidence_surface": "Dame Judith Olivia Dench",
        }
        identity_and_type_mismatch["evidence_proposition"]["subject"] = (
            "Dame Judith Olivia Dench"
        )
        identity_and_type_mismatch["evidence_span"] = (
            "Dame Judith Olivia Dench was born 9 December 1934."
        )
        detail = issue(
            original_question=question,
            artifact=json.dumps(identity_and_type_mismatch),
            tool_receipts=[
                receipt(
                    "date",
                    identity_and_type_mismatch["evidence_span"],
                )
            ],
        )
        self.assertIsNotNone(detail)
        assert detail is not None
        self.assertIn("supplies a date relation argument", detail)
        self.assertNotIn("passage title identity chain", detail)

        predicate_as_entity = json.loads(json.dumps(date_artifact))
        predicate_as_entity["entity_identity"] = {
            "question_surface": "born",
            "evidence_surface": "born",
        }
        predicate_as_entity["target_relation"] = "born"
        predicate_as_entity["evidence_proposition"]["predicate"] = "born"
        detail = issue(
            original_question=question,
            artifact=json.dumps(predicate_as_entity),
            tool_receipts=[receipt("date", date_evidence)],
        )
        self.assertIsNotNone(detail)
        assert detail is not None
        self.assertIn("entity surface must not be the relation predicate", detail)

        place_evidence = (
            "Dench was born in Heworth, North Riding of Yorkshire."
        )
        place_artifact = {
            **date_artifact,
            "target_relation": "was born in",
            "evidence_proposition": {
                "subject": "Dench",
                "predicate": "was born in",
                "object_or_attribute_value": (
                    "Heworth, North Riding of Yorkshire"
                ),
            },
            "evidence_span": place_evidence,
            "passage_id": "place",
        }
        self.assertIsNone(
            issue(
                original_question=question,
                artifact=json.dumps(place_artifact),
                tool_receipts=[receipt("place", place_evidence)],
            )
        )

    def test_evidence_retriever_city_and_date_constraints_are_question_only(
        self,
    ) -> None:
        def validate(
            *, question: str, evidence: str, answer_type: str, value: str
        ) -> str | None:
            passage_id = answer_type
            artifact = {
                "question_scope": question,
                "entity_identity": {
                    "question_surface": "conference",
                    "evidence_surface": "conference",
                },
                "target_relation": "was held in" if answer_type == "location" else "was held on",
                "answer_type_constraint": answer_type,
                "evidence_proposition": {
                    "subject": "conference",
                    "predicate": (
                        "was held in" if answer_type == "location" else "was held on"
                    ),
                    "object_or_attribute_value": value,
                },
                "evidence_span": evidence,
                "passage_id": passage_id,
            }
            receipt = {
                "tool_id": QA_RETRIEVAL_TOOL_ID,
                "tool_version": "frozen-index-v1",
                "request": {
                    "action": "read",
                    "arguments": {"passage_id": passage_id},
                },
                "result": {
                    "value": {
                        "operation": "read",
                        "passage_id": passage_id,
                        "passage": {
                            "passage_id": passage_id,
                            "text": evidence,
                        },
                    },
                    "completed": True,
                },
                "error_type": None,
            }
            return QARetrievalReactExecutionAdapter._evidence_retriever_completion_issue(
                original_question=question,
                artifact=json.dumps(artifact),
                tool_receipts=[receipt],
            )

        self.assertIsNone(
            validate(
                question="What city was the conference held in?",
                evidence="The conference was held in Kyoto.",
                answer_type="location",
                value="Kyoto",
            )
        )
        self.assertIsNone(
            validate(
                question="When was the conference held?",
                evidence="The conference was held on 12 March 2020.",
                answer_type="date",
                value="12 March 2020",
            )
        )
        mismatch = validate(
            question="When was the conference held?",
            evidence="The conference was held on Kyoto.",
            answer_type="date",
            value="Kyoto",
        )
        self.assertIsNotNone(mismatch)
        assert mismatch is not None
        self.assertIn("does not supply the question's requested date", mismatch)

    async def test_wrong_relation_argument_type_advances_retrieval_strategy(
        self,
    ) -> None:
        question = "Where in England was Dame Judi Dench born?"
        evidence = "Dench was born 9 December 1934."
        invalid_artifact = {
            "question_scope": question,
            "entity_identity": {
                "question_surface": "Dench",
                "evidence_surface": "Dench",
            },
            "target_relation": "was born",
            "answer_type_constraint": "location",
            "evidence_proposition": {
                "subject": "Dench",
                "predicate": "was born",
                "object_or_attribute_value": "9 December 1934",
            },
            "evidence_span": evidence,
            "passage_id": "p1",
        }

        def action(name: str, arguments: object) -> str:
            return json.dumps(
                {
                    "arguments": arguments,
                    "kind": "complete" if name == "complete" else "tool",
                    "name": name,
                    "resource_id": (
                        None if name == "complete" else QA_RETRIEVAL_TOOL_ID
                    ),
                    "skill_id": None,
                }
            )

        class DenchDateIndex(FakeIndex):
            def search(self, query: str, *, limit: int) -> tuple[FakeHit, ...]:
                self.search_calls.append((query, limit))
                passage_id = "p1" if len(self.search_calls) == 1 else "p2"
                return (FakeHit(passage_id, passage_id, "Judi Dench", evidence, 1),)

            def read(self, passage_id: str) -> FakePassage:
                self.read_calls.append(passage_id)
                return FakePassage(passage_id, passage_id, "Judi Dench", evidence)

        class SequenceGateway:
            def __init__(self) -> None:
                self.requests: list[AgentRequest] = []
                self.outputs = [
                    action(
                        "search",
                        {"query": "Dame Judi Dench born", "limit": 5},
                    ),
                    action("read", {"passage_id": "p1"}),
                    action("complete", {"value": invalid_artifact}),
                    action(
                        "search",
                        {
                            "query": "Dame Judi Dench birthplace England",
                            "limit": 10,
                        },
                    ),
                ]

            async def generate(self, request: AgentRequest) -> AgentResponse:
                self.requests.append(request)
                return AgentResponse(self.outputs.pop(0))

        index = DenchDateIndex()
        gateway = SequenceGateway()
        with self.assertRaises(ReactExecutionError) as caught:
            await QARetrievalReactExecutionAdapter(
                gateway=gateway,
                tool_registry=build_qa_tool_registry(index),
                max_turns=4,
                max_tool_calls=10,
                task_type="factual_qa",
                completion_policy="required_evidence",
            ).execute(
                AgentRequest(
                    request_id="trivia:dench-answer-type-recovery",
                    run_id="trivia",
                    graph_revision=1,
                    problem=question,
                    agent=AgentNode(
                        "evidence_retriever",
                        "model",
                        "retrieve the requested relation argument",
                        role_family="evidence_retriever",
                        allowed_tools=(QA_RETRIEVAL_TOOL_ID,),
                        execution_mode="react",
                    ),
                    model=ModelSpec("model", "provider"),
                    provider=ProviderSpec("provider", kind="test"),
                    phase=ExecutionPhase.SINGLE,
                    semantic_protocol=QA_VERIFIED_ANSWER_LINEAGE_PROTOCOL,
                )
            )

        self.assertEqual(
            [
                ("Dame Judi Dench born", 5),
                ("Dame Judi Dench birthplace England", 10),
            ],
            index.search_calls,
        )
        feedback = caught.exception.react_trace[2]["public_error_code"]
        self.assertTrue(
            feedback.startswith("qa_semantic_evidence_provenance_invalid:")
        )
        self.assertIn("supplies a date relation argument", feedback)
        self.assertIn("`spelling_normalization`", gateway.requests[3].agent.contract)

    async def test_irrelevant_david_crockett_read_advances_david_soul_retrieval(
        self,
    ) -> None:
        question = "Which city does David Soul come from?"
        crockett_evidence = (
            "David Crockett was born in Greene County, Tennessee."
        )
        invalid_artifact = {
            "question_scope": question,
            "entity_identity": {
                "question_surface": "David Soul",
                "evidence_surface": "David Crockett",
            },
            "target_relation": "was born in",
            "answer_type_constraint": "entity",
            "evidence_proposition": {
                "subject": "David Crockett",
                "predicate": "was born in",
                "object_or_attribute_value": "Greene County, Tennessee",
            },
            "evidence_span": crockett_evidence,
            "passage_id": "p1",
        }

        def action(name: str, arguments: object) -> str:
            return json.dumps(
                {
                    "arguments": arguments,
                    "kind": "complete" if name == "complete" else "tool",
                    "name": name,
                    "resource_id": (
                        None if name == "complete" else QA_RETRIEVAL_TOOL_ID
                    ),
                    "skill_id": None,
                }
            )

        class DisambiguationIndex(FakeIndex):
            def search(self, query: str, *, limit: int) -> tuple[FakeHit, ...]:
                self.search_calls.append((query, limit))
                passage_id = "p1" if len(self.search_calls) == 1 else "p2"
                title = "David Crockett" if passage_id == "p1" else "David Soul"
                return (FakeHit(passage_id, passage_id, title, title, 1),)

            def read(self, passage_id: str) -> FakePassage:
                self.read_calls.append(passage_id)
                return FakePassage(
                    passage_id,
                    passage_id,
                    "David Crockett",
                    crockett_evidence,
                )

        class SequenceGateway:
            def __init__(self) -> None:
                self.outputs = [
                    action("search", {"query": "David Soul birthplace", "limit": 5}),
                    action("read", {"passage_id": "p1"}),
                    action("complete", {"value": invalid_artifact}),
                    action(
                        "search",
                        {"query": "David Soul origin city", "limit": 10},
                    ),
                ]
                self.requests: list[AgentRequest] = []

            async def generate(self, request: AgentRequest) -> AgentResponse:
                self.requests.append(request)
                return AgentResponse(self.outputs.pop(0))

        index = DisambiguationIndex()
        gateway = SequenceGateway()
        request = AgentRequest(
            request_id="trivia:david-soul-retrieval-recovery",
            run_id="trivia",
            graph_revision=1,
            problem=question,
            agent=AgentNode(
                "evidence_retriever",
                "model",
                "retrieve evidence for the original entity and relation",
                role_family="evidence_retriever",
                allowed_tools=(QA_RETRIEVAL_TOOL_ID,),
                execution_mode="react",
            ),
            model=ModelSpec("model", "provider"),
            provider=ProviderSpec("provider", kind="test"),
            phase=ExecutionPhase.SINGLE,
            semantic_protocol=QA_VERIFIED_ANSWER_LINEAGE_PROTOCOL,
        )

        with self.assertRaises(ReactExecutionError) as caught:
            await QARetrievalReactExecutionAdapter(
                gateway=gateway,
                tool_registry=build_qa_tool_registry(index),
                max_turns=4,
                max_tool_calls=10,
                task_type="factual_qa",
                completion_policy="required_evidence",
            ).execute(request)

        self.assertEqual(
            [
                ("David Soul birthplace", 5),
                ("David Soul origin city", 10),
            ],
            index.search_calls,
        )
        invalid_feedback = caught.exception.react_trace[2]["public_error_code"]
        self.assertTrue(
            invalid_feedback.startswith(
                "qa_semantic_evidence_provenance_invalid:"
            )
        )
        self.assertIn("alias identity lacks an explicit binding", invalid_feedback)
        retry_schema = json.loads(
            gateway.requests[3].model.metadata["response_json_schema"]
        )
        self.assertEqual(
            10,
            retry_schema["properties"]["arguments"]["properties"]["limit"][
                "const"
            ],
        )
        self.assertIn("`spelling_normalization`", gateway.requests[3].agent.contract)

    async def test_react_direct_completion_remains_valid_when_dispatch_is_impossible(self) -> None:
        def complete(value: str) -> str:
            return json.dumps(
                {
                    "arguments": {"value": value},
                    "kind": "complete",
                    "name": "complete",
                    "resource_id": None,
                    "skill_id": None,
                }
            )

        class OneShotGateway:
            async def generate(self, request: AgentRequest) -> AgentResponse:
                del request
                return AgentResponse(complete("Ada Lovelace"))

        for allowed_tools, max_tool_calls in (
            ((QA_RETRIEVAL_TOOL_ID,), 0),
            ((), 2),
        ):
            with self.subTest(
                allowed_tools=allowed_tools,
                max_tool_calls=max_tool_calls,
            ):
                response = await QARetrievalReactExecutionAdapter(
                    gateway=OneShotGateway(),
                    tool_registry=build_qa_tool_registry(FakeIndex()),
                    max_turns=1,
                    max_tool_calls=max_tool_calls,
                    completion_policy="optional",
                ).execute(
                    AgentRequest(
                        request_id="qa:no-dispatch",
                        run_id="qa",
                        graph_revision=1,
                        problem="Who wrote the first published algorithm?",
                        agent=AgentNode(
                            "retriever",
                            "model",
                            "answer",
                            allowed_tools=allowed_tools,
                            execution_mode="react",
                        ),
                        model=ModelSpec("model", "provider"),
                        provider=ProviderSpec("provider", kind="test"),
                        phase=ExecutionPhase.SINGLE,
                    )
                )

                self.assertEqual("Ada Lovelace", response.text)
                self.assertEqual(0, response.metadata["tool_calls"])
                self.assertEqual(
                    "completed",
                    response.metadata["react_trace"][0]["observation_status"],
                )


if __name__ == "__main__":
    unittest.main()
