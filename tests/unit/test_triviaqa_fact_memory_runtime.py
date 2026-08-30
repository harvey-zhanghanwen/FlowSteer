from __future__ import annotations

from dataclasses import dataclass, replace
import json
from types import SimpleNamespace
import unittest

from src.interactive.agent_graph import AgentGraph, AgentNode, AgentRelation
from src.interactive.agent_runtime import (
    AgentRequest,
    ExecutionPhase,
    UpstreamMessage,
)
from src.interactive.agent_workflow_env import AgentWorkflowEnv
from src.interactive.model_registry import ModelSpec, ProviderSpec
from src.interactive.openai_gateway import build_agent_messages
from src.interactive.qa_tool_adapter import (
    QARetrievalReactExecutionAdapter,
    QA_VERIFIED_ANSWER_LINEAGE_PROTOCOL,
    TRIVIAQA_QA_MEMORY_TOOL_ID,
    _missing_question_named_constraints,
    _missing_question_retrieval_named_constraints,
    _question_retrieval_entity_anchor_alternatives,
    build_qa_tool_registry,
    qa_reasoner_artifact_json_schema,
)
from src.interactive.tool_runtime import ActionKind, StructuredAction, ToolRequest


@dataclass(frozen=True)
class FactManifest:
    corpus_name: str = "triviaqa-semantic-fact-memory"
    corpus_version: str = "v1"
    index_id: str = "triviaqa-fact-memory-v1"
    format: str = "flowsteer.triviaqa.fact-memory-embedding-index.v1"
    retrieval_backend: str = "bge-cosine"
    tool_id: str = TRIVIAQA_QA_MEMORY_TOOL_ID
    record_kind: str = "fact_memory"
    frozen_top_k: int = 2


@dataclass(frozen=True)
class FactHit:
    memory_id: str
    rank: int
    similarity: float
    fact_text: str


@dataclass(frozen=True)
class FactRecord:
    schema_version: str
    memory_id: str
    tool_id: str
    fact_text: str


class FactIndex:
    manifest = FactManifest()

    def __init__(self) -> None:
        self.records = {
            "fact-1": FactRecord(
                "flowsteer.triviaqa.fact_memory.record.v1",
                "fact-1",
                TRIVIAQA_QA_MEMORY_TOOL_ID,
                "Ada Lovelace wrote the first published algorithm.",
            ),
            "fact-2": FactRecord(
                "flowsteer.triviaqa.fact_memory.record.v1",
                "fact-2",
                TRIVIAQA_QA_MEMORY_TOOL_ID,
                "Charles Babbage designed the Analytical Engine.",
            ),
        }

    def search(self, query: str, *, limit: int) -> tuple[FactHit, ...]:
        del query
        return tuple(
            FactHit(record.memory_id, rank, 1.0 - rank / 10, record.fact_text)
            for rank, record in enumerate(self.records.values(), start=1)
        )[:limit]

    def read(self, memory_id: str) -> FactRecord:
        return self.records[memory_id]

    def close(self) -> None:
        return None


def _fact_receipts() -> tuple[dict[str, object], ...]:
    hits = [
        {
            "memory_id": "fact-1",
            "rank": 1,
            "similarity": 0.9,
            "fact_text": "Ada Lovelace wrote the first published algorithm.",
        },
        {
            "memory_id": "fact-2",
            "rank": 2,
            "similarity": 0.8,
            "fact_text": "Charles Babbage designed the Analytical Engine.",
        },
    ]
    search = {
        "tool_id": TRIVIAQA_QA_MEMORY_TOOL_ID,
        "error_type": None,
        "request": {
            "action": "search",
            "arguments": {"query": "first published algorithm", "limit": 2},
        },
        "result": {
            "completed": True,
            "value": {
                "operation": "search",
                "query": "first published algorithm",
                "top_k": 2,
                "memory_ids": ["fact-1", "fact-2"],
                "hits": hits,
            },
        },
    }
    reads = tuple(
        {
            "tool_id": TRIVIAQA_QA_MEMORY_TOOL_ID,
            "error_type": None,
            "request": {
                "action": "read",
                "arguments": {"memory_id": hit["memory_id"]},
            },
            "result": {
                "completed": True,
                "value": {
                    "operation": "read",
                    "memory_id": hit["memory_id"],
                    "memory": {
                        "memory_id": hit["memory_id"],
                        "fact_text": hit["fact_text"],
                    },
                },
            },
        }
        for hit in hits
    )
    return (search, *reads)


def _reasoner_request(artifact: str, receipts: tuple[dict[str, object], ...]) -> AgentRequest:
    return AgentRequest(
        request_id="triviaqa:fact-memory",
        run_id="fact-memory",
        graph_revision=2,
        problem="Who wrote the first published algorithm?",
        agent=AgentNode(
            "reasoner",
            "model",
            "derive the semantic answer from routed evidence",
            role_family="reasoner",
        ),
        model=ModelSpec("model", "provider"),
        provider=ProviderSpec("provider", kind="test"),
        phase=ExecutionPhase.SINGLE,
        semantic_protocol=QA_VERIFIED_ANSWER_LINEAGE_PROTOCOL,
        upstream=(
            UpstreamMessage(
                "retriever",
                "reasoner",
                artifact,
                graph_revision=2,
                artifact_type="evidence",
                artifact_version="artifact-v1",
                tool_receipts=receipts,
            ),
        ),
    )


class TriviaQAFactMemoryRuntimeTests(unittest.IsolatedAsyncioTestCase):
    def test_coordinated_entity_query_decomposition_preserves_final_scope(
        self,
    ) -> None:
        question = (
            "Who wrote The Turn Of The Screw in the 19th century and "
            "The Ambassadors in the 20th?"
        )
        self.assertEqual(
            (("turn", "of", "the", "screw"), ("ambassadors",)),
            _question_retrieval_entity_anchor_alternatives(question),
        )
        self.assertEqual(
            (),
            _missing_question_retrieval_named_constraints(
                question,
                "The Ambassadors wrote 20th century",
            ),
        )
        self.assertEqual(
            ("screw", "turn"),
            _missing_question_named_constraints(
                question,
                "The Ambassadors wrote 20th century",
            ),
        )

        adapter = QARetrievalReactExecutionAdapter(
            gateway=SimpleNamespace(generate=lambda request: None),
            tool_registry=build_qa_tool_registry(FactIndex()),
            max_turns=8,
            max_tool_calls=6,
            task_type="factual_qa",
            completion_policy="required_evidence",
            parametric_fallback_after_coverage_failure=True,
            retrieval_tool_id=TRIVIAQA_QA_MEMORY_TOOL_ID,
        )
        request = AgentRequest(
            request_id="triviaqa:coordinated-entity",
            run_id="fact-memory",
            graph_revision=1,
            problem=question,
            agent=AgentNode(
                "retriever",
                "model",
                "retrieve evidence",
                role_family="evidence_retriever",
                allowed_tools=(TRIVIAQA_QA_MEMORY_TOOL_ID,),
                execution_mode="react",
            ),
            model=ModelSpec("model", "provider"),
            provider=ProviderSpec("provider", kind="test"),
            phase=ExecutionPhase.SINGLE,
            semantic_protocol=QA_VERIFIED_ANSWER_LINEAGE_PROTOCOL,
        )
        public_state = adapter._public_retrieval_continuation_state(
            request,
            [],
        )
        assert public_state is not None
        self.assertEqual(
            [["turn", "of", "the", "screw"], ["ambassadors"]],
            public_state["question_entity_anchor_alternatives"],
        )
        for query in (
            "The Turn Of The Screw wrote 19th century",
            "The Ambassadors wrote 20th century",
        ):
            with self.subTest(query=query):
                issue = adapter._tool_action_error(
                    request=request,
                    action=StructuredAction(
                        ActionKind.TOOL,
                        "search",
                        {"query": query, "limit": 2},
                        resource_id=TRIVIAQA_QA_MEMORY_TOOL_ID,
                    ),
                    observations=[],
                )
                self.assertIsNone(issue)
        issue = adapter._tool_action_error(
            request=request,
            action=StructuredAction(
                ActionKind.TOOL,
                "search",
                {"query": "wrote 19th century", "limit": 2},
                resource_id=TRIVIAQA_QA_MEMORY_TOOL_ID,
            ),
            observations=[],
        )
        self.assertEqual("qa_retrieval_query_entity_anchor_loss", issue)

    async def test_registry_projects_fact_only_search_and_read_payloads(self) -> None:
        registry = build_qa_tool_registry(FactIndex())
        self.assertEqual((TRIVIAQA_QA_MEMORY_TOOL_ID,), registry.resource_ids)
        capability = registry.require_capability(TRIVIAQA_QA_MEMORY_TOOL_ID)
        self.assertEqual(
            "fact_memory",
            capability.output_schema["x-flowsteer-record-kind"],
        )

        result, receipt = await registry.ainvoke_with_receipt(
            TRIVIAQA_QA_MEMORY_TOOL_ID,
            ToolRequest(
                "search",
                {"query": "first published algorithm", "limit": 2},
            ),
        )
        self.assertIsNone(receipt.error_type)
        assert result is not None
        self.assertEqual(["fact-1", "fact-2"], result.value["memory_ids"])
        self.assertNotIn("passage_ids", result.value)
        self.assertEqual(
            {"memory_id", "rank", "similarity", "fact_text"},
            set(result.value["hits"][0]),
        )

        result, receipt = await registry.ainvoke_with_receipt(
            TRIVIAQA_QA_MEMORY_TOOL_ID,
            ToolRequest("read", {"memory_id": "fact-1"}),
        )
        self.assertIsNone(receipt.error_type)
        assert result is not None
        self.assertEqual(
            {"memory_id", "fact_text"},
            set(result.value["memory"]),
        )
        self.assertNotIn("canonical_answer", json.dumps(result.value))
        self.assertNotIn("source_train_task_id", json.dumps(result.value))

    def test_fact_projection_requires_rank_ordered_read_for_every_topk_hit(self) -> None:
        receipts = _fact_receipts()
        selection = json.dumps(
            {
                "retrieval_status": "evidence_found",
                "relevant_ranks": [1],
            }
        )
        projected, issue = (
            QARetrievalReactExecutionAdapter._qa_memory_completion_receipt_projection(
                original_question="Who wrote the first published algorithm?",
                selection_artifact=selection,
                tool_receipts=receipts,
                parametric_fallback_after_coverage_failure=True,
                expected_source_task_id="must-not-enter-fact-profile",
            )
        )
        self.assertIsNone(issue)
        assert projected is not None
        artifact = json.loads(projected)
        self.assertEqual(["fact-1", "fact-2"], [c["memory_id"] for c in artifact["candidates"]])
        self.assertEqual(["fact-1"], artifact["relevant_memory_ids"])
        self.assertTrue(all(set(c) == {"memory_id", "rank", "similarity", "fact_text"} for c in artifact["candidates"]))
        self.assertNotIn("canonical_answer", projected)
        self.assertNotIn("source_train_task_id", projected)

        _, issue = QARetrievalReactExecutionAdapter._qa_memory_completion_receipt_projection(
            original_question="Who wrote the first published algorithm?",
            selection_artifact=selection,
            tool_receipts=(receipts[0], receipts[2], receipts[1]),
            parametric_fallback_after_coverage_failure=True,
        )
        self.assertIn("original rank order", issue or "")

        for ranks in ([2, 1], [1, 1], [3]):
            with self.subTest(relevant_ranks=ranks):
                _, issue = QARetrievalReactExecutionAdapter._qa_memory_completion_receipt_projection(
                    original_question="Who wrote the first published algorithm?",
                    selection_artifact=json.dumps(
                        {
                            "retrieval_status": "evidence_found",
                            "relevant_ranks": ranks,
                        }
                    ),
                    tool_receipts=receipts,
                    parametric_fallback_after_coverage_failure=True,
                )
                self.assertIsNotNone(issue)

    def test_ranked_action_domain_reads_every_fact_before_completion(self) -> None:
        registry = build_qa_tool_registry(FactIndex())
        adapter = QARetrievalReactExecutionAdapter(
            gateway=SimpleNamespace(generate=lambda request: None),
            tool_registry=registry,
            max_turns=8,
            max_tool_calls=4,
            task_type="factual_qa",
            completion_policy="required_evidence",
            parametric_fallback_after_coverage_failure=True,
            retrieval_tool_id=TRIVIAQA_QA_MEMORY_TOOL_ID,
        )
        request = AgentRequest(
            request_id="triviaqa:worker",
            run_id="fact-memory",
            graph_revision=1,
            problem="Who wrote the first published algorithm?",
            agent=AgentNode(
                "retriever",
                "model",
                "retrieve evidence",
                role_family="evidence_retriever",
                allowed_tools=(TRIVIAQA_QA_MEMORY_TOOL_ID,),
                execution_mode="react",
            ),
            model=ModelSpec("model", "provider"),
            provider=ProviderSpec("provider", kind="test"),
            phase=ExecutionPhase.SINGLE,
            semantic_protocol=QA_VERIFIED_ANSWER_LINEAGE_PROTOCOL,
        )
        completion_schema = adapter._completion_arguments_schema(request)
        value_schema = completion_schema["properties"]["value"]
        self.assertEqual(
            ["retrieval_status", "relevant_ranks"],
            value_schema["required"],
        )
        self.assertNotIn("memory_ids", value_schema["properties"])
        self.assertNotIn("relevant_memory_ids", value_schema["properties"])
        receipts = _fact_receipts()
        observations = []
        for receipt in receipts:
            observations.append(
                {
                    "observation_status": "success",
                    "executed_action": {
                        "kind": "tool",
                        "name": receipt["request"]["action"],
                        "resource_id": TRIVIAQA_QA_MEMORY_TOOL_ID,
                        "arguments": receipt["request"]["arguments"],
                    },
                    "result": receipt["result"]["value"],
                }
            )
        actions, complete = adapter._state_conditioned_action_domain(
            request,
            observations[:1],
        )
        self.assertEqual(
            frozenset({(TRIVIAQA_QA_MEMORY_TOOL_ID, "read")}),
            actions,
        )
        self.assertFalse(complete)
        _, complete = adapter._state_conditioned_action_domain(
            request,
            observations[:2],
        )
        self.assertFalse(complete)
        actions, complete = adapter._state_conditioned_action_domain(
            request,
            observations,
        )
        self.assertEqual(frozenset(), actions)
        self.assertTrue(complete)

    def test_fact_read_is_receipt_grounded_and_extra_answer_field_is_rejected(self) -> None:
        receipt = _fact_receipts()[1]
        self.assertTrue(
            AgentWorkflowEnv._successful_read_receipt(
                receipt,
                TRIVIAQA_QA_MEMORY_TOOL_ID,
            )
        )
        self.assertEqual(
            "Ada Lovelace wrote the first published algorithm.",
            AgentWorkflowEnv._successful_read_text(
                receipt,
                TRIVIAQA_QA_MEMORY_TOOL_ID,
            ),
        )
        leaked = json.loads(json.dumps(receipt))
        leaked["result"]["value"]["memory"]["canonical_answer"] = "Ada Lovelace"
        self.assertFalse(
            AgentWorkflowEnv._successful_read_receipt(
                leaked,
                TRIVIAQA_QA_MEMORY_TOOL_ID,
            )
        )

    def test_react_adapter_admits_canonical_fact_read_receipt(self) -> None:
        receipt = _fact_receipts()[1]

        self.assertTrue(
            QARetrievalReactExecutionAdapter._successful_read_receipt(
                receipt,
                TRIVIAQA_QA_MEMORY_TOOL_ID,
            )
        )
        self.assertEqual(
            "Ada Lovelace wrote the first published algorithm.",
            QARetrievalReactExecutionAdapter._successful_read_text(
                receipt,
                TRIVIAQA_QA_MEMORY_TOOL_ID,
            ),
        )

        leaked = json.loads(json.dumps(receipt))
        leaked["result"]["value"]["memory"]["canonical_answer"] = "Ada Lovelace"
        self.assertFalse(
            QARetrievalReactExecutionAdapter._successful_read_receipt(
                leaked,
                TRIVIAQA_QA_MEMORY_TOOL_ID,
            )
        )

    def test_reasoner_schema_and_prompt_have_no_canonical_answer_constraint(self) -> None:
        projected, issue = (
            QARetrievalReactExecutionAdapter._qa_memory_completion_receipt_projection(
                original_question="Who wrote the first published algorithm?",
                selection_artifact=json.dumps(
                    {
                        "retrieval_status": "evidence_found",
                        "relevant_ranks": [1],
                    }
                ),
                tool_receipts=_fact_receipts(),
                parametric_fallback_after_coverage_failure=True,
            )
        )
        self.assertIsNone(issue)
        assert projected is not None
        request = _reasoner_request(projected, _fact_receipts())
        schema = qa_reasoner_artifact_json_schema(request)
        assert schema is not None
        candidate_schema = schema["properties"]["candidate_answer"]
        self.assertNotIn("const", candidate_schema)
        proposition_schema = schema["properties"]["evidence_propositions"]["items"]
        self.assertNotIn("const", proposition_schema["properties"]["subject"])
        messages = build_agent_messages(request)
        self.assertIn("self-contained declarative fact_text", messages[0]["content"])
        self.assertIn("hidden canonical answer", messages[0]["content"])

        misrouted = replace(
            request,
            upstream=(
                UpstreamMessage(
                    "retriever",
                    "different-agent",
                    projected,
                    graph_revision=2,
                    artifact_type="evidence",
                    artifact_version="artifact-v1",
                    tool_receipts=_fact_receipts(),
                ),
            ),
        )
        self.assertNotIn(
            "self-contained declarative fact_text",
            build_agent_messages(misrouted)[0]["content"],
        )

    def test_reasoner_repairs_answer_slot_from_model_authored_fact(self) -> None:
        base = {
            "candidate_answer": "Ada Lovelace",
            "answer_slot": {
                "proposition_index": 0,
                "answer_field": "object_or_attribute_value",
            },
            "evidence_propositions": [
                {
                    "subject": "Ada Lovelace",
                    "relation": "wrote",
                    "object_or_attribute_value": "the first published algorithm",
                    "evidence_span": (
                        "Ada Lovelace wrote the first published algorithm."
                    ),
                }
            ],
        }
        constraints = (
            QARetrievalReactExecutionAdapter._reasoner_repair_constraints(
                base,
                (
                    "Reasoner answer_slot.answer_field selects a proposition "
                    "argument that does not match candidate_answer exactly"
                ),
            )
        )
        self.assertEqual(0, constraints[("answer_slot", "proposition_index")])
        self.assertEqual("subject", constraints[("answer_slot", "answer_field")])

        outside = json.loads(json.dumps(base))
        outside["candidate_answer"] = "the first published algorithm"
        outside["answer_slot"]["proposition_index"] = 1
        constraints = (
            QARetrievalReactExecutionAdapter._reasoner_repair_constraints(
                outside,
                "Reasoner answer_slot.proposition_index is outside evidence_propositions",
            )
        )
        self.assertEqual(0, constraints[("answer_slot", "proposition_index")])
        self.assertEqual(
            "object_or_attribute_value",
            constraints[("answer_slot", "answer_field")],
        )

        subspan = json.loads(json.dumps(base))
        subspan["candidate_answer"] = "1914"
        subspan["answer_slot"]["answer_field"] = "object_or_attribute_value"
        subspan["evidence_propositions"][0]["object_or_attribute_value"] = (
            "the year 1914"
        )
        subspan["evidence_propositions"][0]["evidence_span"] = (
            "The event began in the year 1914."
        )
        constraints = (
            QARetrievalReactExecutionAdapter._reasoner_repair_constraints(
                subspan,
                (
                    "Reasoner candidate_answer must copy the proposition argument "
                    "selected by answer_slot exactly"
                ),
            )
        )
        self.assertEqual(
            "1914",
            constraints[
                (
                    "evidence_propositions",
                    "0",
                    "object_or_attribute_value",
                )
            ],
        )

        duplicate = {
            "candidate_answer": "Norway",
            "answer_slot": {
                "proposition_index": 0,
                "answer_field": "subject",
            },
            "evidence_propositions": [
                {
                    "subject": "Norway",
                    "relation": "was",
                    "object_or_attribute_value": "Norway",
                    "evidence_span": (
                        "Norway was the first European country to abolish "
                        "capital punishment."
                    ),
                }
            ],
        }
        constraints = (
            QARetrievalReactExecutionAdapter._reasoner_repair_constraints(
                duplicate,
                (
                    "Reasoner evidence_propositions[0] must bind distinct subject "
                    "and object_or_attribute_value arguments"
                ),
            )
        )
        self.assertEqual(
            "the first European country to abolish capital punishment",
            constraints[
                (
                    "evidence_propositions",
                    "0",
                    "object_or_attribute_value",
                )
            ],
        )

    def test_fact_only_receipt_binds_synonymous_question_relation(self) -> None:
        question = "What is Patricia Neary famous in?"
        fact_text = "Patricia Neary is renowned in Ballet."
        receipt = json.loads(json.dumps(_fact_receipts()[1]))
        receipt["request"]["arguments"]["memory_id"] = "fact-neary"
        receipt["result"]["value"]["memory_id"] = "fact-neary"
        receipt["result"]["value"]["memory"] = {
            "memory_id": "fact-neary",
            "fact_text": fact_text,
        }
        read_text = AgentWorkflowEnv._successful_read_text(
            receipt,
            TRIVIAQA_QA_MEMORY_TOOL_ID,
        )
        self.assertIsNotNone(read_text)
        assert read_text is not None
        artifact = json.dumps(
            {
                "question_scope": question,
                "answer_slot": {
                    "answer_type": "short_answer",
                    "answer_cardinality": "single",
                    "qualifiers": [],
                    "proposition_index": 0,
                    "answer_field": "object_or_attribute_value",
                },
                "evidence_propositions": [
                    {
                        "subject": "Patricia Neary",
                        "relation": "is renowned in",
                        "object_or_attribute_value": "Ballet",
                        "qualifiers": [],
                        "evidence_span": fact_text,
                    }
                ],
                "multi_hop_chain": [fact_text],
                "candidate_answer": "Ballet",
                "evidence": [fact_text],
            }
        )
        self.assertIsNone(
            AgentWorkflowEnv._reasoner_evidence_provenance_issue(
                artifact,
                [read_text],
                require_answer_binding=True,
                original_question=question,
                qa_memory_relevant_memory_ids=["fact-neary"],
            )
        )
        self.assertIsNotNone(
            AgentWorkflowEnv._reasoner_evidence_provenance_issue(
                artifact,
                [read_text],
                require_answer_binding=True,
                original_question=question,
                qa_memory_relevant_memory_ids=["different-memory"],
            )
        )
        wrong_candidate = json.loads(artifact)
        wrong_candidate["candidate_answer"] = "Opera"
        self.assertIsNotNone(
            AgentWorkflowEnv._reasoner_evidence_provenance_issue(
                json.dumps(wrong_candidate),
                [read_text],
                require_answer_binding=True,
                original_question=question,
                qa_memory_relevant_memory_ids=["fact-neary"],
            )
        )

    def test_fact_only_binding_normalizes_unique_slot_and_relation_surface(self) -> None:
        question = "The VS-300 was a type of what?"
        fact_text = "The classification of the VS-300 is Helicopter."
        receipt = json.loads(json.dumps(_fact_receipts()[1]))
        receipt["request"]["arguments"]["memory_id"] = "fact-vs300"
        receipt["result"]["value"]["memory_id"] = "fact-vs300"
        receipt["result"]["value"]["memory"] = {
            "memory_id": "fact-vs300",
            "fact_text": fact_text,
        }
        read_text = AgentWorkflowEnv._successful_read_text(
            receipt,
            TRIVIAQA_QA_MEMORY_TOOL_ID,
        )
        self.assertIsNotNone(read_text)
        assert read_text is not None
        artifact = json.dumps(
            {
                "question_scope": question,
                "answer_slot": {
                    "answer_type": "short_answer",
                    "answer_cardinality": "single",
                    "qualifiers": [],
                    "proposition_index": 0,
                    "answer_field": "subject",
                },
                "evidence_propositions": [
                    {
                        "subject": "The VS-300",
                        "relation": "is a type of",
                        "object_or_attribute_value": "Helicopter",
                        "qualifiers": [],
                        "evidence_span": fact_text,
                    }
                ],
                "multi_hop_chain": [fact_text],
                "candidate_answer": "Helicopter",
                "evidence": [fact_text],
            }
        )
        candidate, issue = AgentWorkflowEnv._reasoner_candidate(
            artifact,
            original_question=question,
            minimum_evidence_propositions=1,
            minimum_reasoning_steps=1,
            allow_fact_memory_binding=True,
        )
        self.assertIsNone(issue)
        self.assertEqual("Helicopter", candidate)
        self.assertIsNone(
            AgentWorkflowEnv._reasoner_evidence_provenance_issue(
                artifact,
                [read_text],
                require_answer_binding=True,
                original_question=question,
                qa_memory_relevant_memory_ids=["fact-vs300"],
            )
        )

    def test_fact_only_binding_admits_requested_numeric_span(self) -> None:
        question = (
            "When did the founder of Jehovah's Witnesses say the world would end?"
        )
        fact_text = (
            "The founder of Jehovah's Witnesses stated that the world would "
            "come to an end in 1914."
        )
        receipt = json.loads(json.dumps(_fact_receipts()[1]))
        receipt["request"]["arguments"]["memory_id"] = "fact-1914"
        receipt["result"]["value"]["memory_id"] = "fact-1914"
        receipt["result"]["value"]["memory"] = {
            "memory_id": "fact-1914",
            "fact_text": fact_text,
        }
        read_text = AgentWorkflowEnv._successful_read_text(
            receipt,
            TRIVIAQA_QA_MEMORY_TOOL_ID,
        )
        self.assertIsNotNone(read_text)
        assert read_text is not None
        artifact = json.dumps(
            {
                "question_scope": question,
                "answer_slot": {
                    "answer_type": "date",
                    "answer_cardinality": "single",
                    "qualifiers": [],
                    "proposition_index": 0,
                    "answer_field": "object_or_attribute_value",
                },
                "evidence_propositions": [
                    {
                        "subject": "The founder of Jehovah's Witnesses",
                        "relation": "stated",
                        "object_or_attribute_value": (
                            "the world would come to an end in 1914"
                        ),
                        "qualifiers": [],
                        "evidence_span": fact_text,
                    }
                ],
                "multi_hop_chain": [fact_text],
                "candidate_answer": "1914",
                "evidence": [fact_text],
            }
        )
        _, strict_issue = AgentWorkflowEnv._reasoner_candidate(
            artifact,
            original_question=question,
            minimum_evidence_propositions=1,
            minimum_reasoning_steps=1,
        )
        self.assertIn("must copy the proposition argument", strict_issue or "")
        candidate, issue = AgentWorkflowEnv._reasoner_candidate(
            artifact,
            original_question=question,
            minimum_evidence_propositions=1,
            minimum_reasoning_steps=1,
            allow_fact_memory_binding=True,
        )
        self.assertIsNone(issue)
        self.assertEqual("1914", candidate)
        self.assertIsNone(
            AgentWorkflowEnv._reasoner_evidence_provenance_issue(
                artifact,
                [read_text],
                require_answer_binding=True,
                original_question=question,
                qa_memory_relevant_memory_ids=["fact-1914"],
            )
        )

    def test_fact_artifact_ingress_requires_explicit_direct_relation(self) -> None:
        projected, issue = (
            QARetrievalReactExecutionAdapter._qa_memory_completion_receipt_projection(
                original_question="Who wrote the first published algorithm?",
                selection_artifact=json.dumps(
                    {
                        "retrieval_status": "evidence_found",
                        "relevant_ranks": [1],
                    }
                ),
                tool_receipts=_fact_receipts(),
                parametric_fallback_after_coverage_failure=True,
            )
        )
        self.assertIsNone(issue)
        assert projected is not None
        retriever = AgentNode(
            "retriever",
            "model",
            "retrieve evidence",
            role_family="evidence_retriever",
            allowed_tools=(TRIVIAQA_QA_MEMORY_TOOL_ID,),
            execution_mode="react",
        )
        reasoner = AgentNode(
            "reasoner",
            "model",
            "derive answer",
            role_family="reasoner",
        )
        env = object.__new__(AgentWorkflowEnv)
        env.semantic_protocol = QA_VERIFIED_ANSWER_LINEAGE_PROTOCOL
        env.required_evidence_tool_id = TRIVIAQA_QA_MEMORY_TOOL_ID
        env.parametric_fallback_after_coverage_failure = True
        env._problem = "Who wrote the first published algorithm?"
        env._graph = AgentGraph((retriever, reasoner))
        outputs = {"retriever": projected}
        metadata = {
            "retriever": {
                "artifact_version": "artifact-v1",
                "tool_receipts": list(_fact_receipts()),
            },
            "reasoner": {
                "input_artifact_versions": {"retriever": "artifact-v1"},
                "input_artifact_provenance": [
                    {
                        "source_agent_id": "retriever",
                        "target_agent_id": "reasoner",
                        "artifact_version": "artifact-v1",
                        "artifact": projected,
                        "tool_receipts": list(_fact_receipts()),
                    }
                ],
            },
        }
        self.assertIn(
            "explicit direct",
            env._triviaqa_qa_memory_ingress_issue(
                outputs,
                metadata,
                retriever_id="retriever",
                reasoner_id="reasoner",
            )
            or "",
        )
        env._graph = AgentGraph(
            (retriever, reasoner),
            (AgentRelation("retriever", "reasoner", True, False),),
        )
        self.assertIsNone(
            env._triviaqa_qa_memory_ingress_issue(
                outputs,
                metadata,
                retriever_id="retriever",
                reasoner_id="reasoner",
            )
        )


if __name__ == "__main__":
    unittest.main()
