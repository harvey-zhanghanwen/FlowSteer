from __future__ import annotations

from dataclasses import dataclass, replace
import json
from types import SimpleNamespace
import unittest

from src.interactive.agent_graph import AgentNode
from src.interactive.agent_workflow_env import AgentWorkflowEnv
from src.interactive.agent_runtime import AgentRequest, AgentResponse, ExecutionPhase
from src.interactive.model_registry import ModelSpec, ProviderSpec
from src.interactive.qa_tool_adapter import (
    QA_VERIFIED_ANSWER_LINEAGE_PROTOCOL,
    QARetrievalReactExecutionAdapter,
    TRIVIAQA_QA_MEMORY_TOOL_ID,
    build_qa_tool_registry,
)
from src.interactive.tool_runtime import ToolRequest


@dataclass(frozen=True)
class _Manifest:
    corpus_name: str = "triviaqa-frozen-train-qa-memory"
    corpus_version: str = "train-memory-v1"
    index_id: str = "qa-memory-index-v1"
    format: str = "flowsteer.triviaqa.qa-memory-embedding-index.v1"
    retrieval_backend: str = "normalized-dot-product"
    tool_id: str = TRIVIAQA_QA_MEMORY_TOOL_ID
    frozen_top_k: int = 3
    tool_budget: object = None


@dataclass(frozen=True)
class _MemoryHit:
    passage_id: str = "memory-001"
    document_id: str = "triviaqa:tc_129"
    title: str = "Which author wrote the novel?"
    snippet: str = "Question: Which author wrote the novel? Answer: The answer is Ada"
    rank: int = 1
    similarity: float = 0.93
    memory_id: str = "memory-001"
    source_train_task_id: str = "triviaqa:tc_129"
    base_task_id: str = "triviaqa:tc_129"
    cycled_training_sample: bool = False
    cycle_index: int | None = None
    paraphrase_question: str = "Which author wrote the novel?"
    paraphrase_answer_statement: str = "The answer is Ada"
    canonical_answer: str = "Ada"
    paraphrase_version: str = "semantic-preserving-v1"
    paraphrase_provenance: object = None


@dataclass(frozen=True)
class _MemoryRecord:
    passage_id: str = "memory-001"
    document_id: str = "triviaqa:tc_129"
    title: str = "Which author wrote the novel?"
    text: str = "Question: Which author wrote the novel?\nAnswer: The answer is Ada"
    memory_id: str = "memory-001"
    source_train_task_id: str = "triviaqa:tc_129"
    base_task_id: str = "triviaqa:tc_129"
    cycled_training_sample: bool = False
    cycle_index: int | None = None
    paraphrase_question: str = "Which author wrote the novel?"
    paraphrase_answer_statement: str = "The answer is Ada"
    canonical_answer: str = "Ada"
    paraphrase_version: str = "semantic-preserving-v1"
    paraphrase_provenance: object = None


class _Index:
    manifest = _Manifest()

    def __init__(self) -> None:
        self.search_calls: list[tuple[str, int]] = []
        self.read_calls: list[str] = []

    def search(self, query: str, *, limit: int) -> tuple[_MemoryHit, ...]:
        self.search_calls.append((query, limit))
        return (_MemoryHit(),)

    def read(self, memory_id: str) -> _MemoryRecord:
        self.read_calls.append(memory_id)
        return _MemoryRecord()

    def close(self) -> None:
        return None


class TriviaQAQAMemoryV2AdapterTests(unittest.IsolatedAsyncioTestCase):
    def _request(self) -> AgentRequest:
        return AgentRequest(
            request_id="triviaqa:qamemory-adapter",
            run_id="triviaqa-qamemory-v2",
            graph_revision=1,
            problem="Which author wrote the novel?",
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
        )

    async def test_registry_uses_canonical_qamemory_wire_and_receipts(self) -> None:
        index = _Index()
        registry = build_qa_tool_registry(index)

        self.assertEqual((TRIVIAQA_QA_MEMORY_TOOL_ID,), registry.resource_ids)
        capability = registry.require_capability(TRIVIAQA_QA_MEMORY_TOOL_ID)
        self.assertEqual({"const": 3, "description": capability.action_schemas["search"]["properties"]["limit"]["description"]}, capability.action_schemas["search"]["properties"]["limit"])
        self.assertEqual(
            ["memory_id"],
            capability.action_schemas["read"]["required"],
        )

        search_result, search_receipt = await registry.ainvoke_with_receipt(
            TRIVIAQA_QA_MEMORY_TOOL_ID,
            ToolRequest("search", {"query": "author novel", "limit": 3}),
        )
        self.assertIsNotNone(search_result)
        assert search_result is not None
        self.assertEqual(["memory-001"], search_result.value["memory_ids"])
        self.assertEqual(["memory-001"], search_result.value["passage_ids"])
        self.assertEqual(TRIVIAQA_QA_MEMORY_TOOL_ID, search_receipt.tool_id)

        read_result, read_receipt = await registry.ainvoke_with_receipt(
            TRIVIAQA_QA_MEMORY_TOOL_ID,
            ToolRequest("read", {"memory_id": "memory-001"}),
        )
        self.assertIsNotNone(read_result)
        assert read_result is not None
        self.assertEqual("memory-001", read_result.value["memory_id"])
        self.assertEqual("memory-001", read_result.value["passage_id"])
        self.assertEqual(read_result.value["memory"], read_result.value["passage"])
        self.assertEqual(TRIVIAQA_QA_MEMORY_TOOL_ID, read_receipt.tool_id)
        self.assertEqual(
            {"memory_id": "memory-001"},
            dict(read_receipt.request.arguments),
        )
        self.assertTrue(
            QARetrievalReactExecutionAdapter._successful_read_receipt(
                read_receipt.to_value(),
                TRIVIAQA_QA_MEMORY_TOOL_ID,
            )
        )

        serialized = str(search_result.value) + str(read_result.value)
        for forbidden in (
            "validation_question",
            "validation_answer",
            "accepted_answers",
            "evaluator_receipt",
            "supporting_facts",
        ):
            self.assertNotIn(forbidden, serialized)

    async def test_full_react_execution_keeps_worker_owned_qamemory_receipts(
        self,
    ) -> None:
        def action(name: str, arguments: object) -> str:
            import json

            return json.dumps(
                {
                    "arguments": arguments,
                    "kind": "complete" if name == "complete" else "tool",
                    "name": name,
                    "resource_id": (
                        None
                        if name == "complete"
                        else TRIVIAQA_QA_MEMORY_TOOL_ID
                    ),
                    "skill_id": None,
                }
            )

        class _Gateway:
            def __init__(self) -> None:
                self.outputs = [
                    action("search", {"query": "author novel", "limit": 3}),
                    action("read", {"memory_id": "memory-001"}),
                    action("complete", {"value": "Ada"}),
                ]

            async def generate(self, request: AgentRequest) -> AgentResponse:
                self.assert_worker_request(request)
                return AgentResponse(self.outputs.pop(0))

            @staticmethod
            def assert_worker_request(request: AgentRequest) -> None:
                if request.agent.allowed_tools != (
                    TRIVIAQA_QA_MEMORY_TOOL_ID,
                ):
                    raise AssertionError("QA-memory Tool must remain worker-owned")

        index = _Index()
        adapter = QARetrievalReactExecutionAdapter(
            gateway=_Gateway(),
            tool_registry=build_qa_tool_registry(index),
            max_turns=3,
            max_tool_calls=2,
            task_type=None,
            completion_policy="required_evidence",
            retrieval_tool_id=TRIVIAQA_QA_MEMORY_TOOL_ID,
        )
        response = await adapter.execute(self._request())

        self.assertEqual("Ada", response.text)
        receipts = response.metadata["tool_receipts"]
        self.assertEqual(2, len(receipts))
        self.assertTrue(
            all(
                receipt["tool_id"] == TRIVIAQA_QA_MEMORY_TOOL_ID
                for receipt in receipts
            )
        )
        self.assertEqual(
            {"memory_id": "memory-001"},
            receipts[1]["request"]["arguments"],
        )
        self.assertEqual([("author novel", 3)], index.search_calls)
        self.assertEqual(["memory-001"], index.read_calls)

    def test_v63_state_conditioned_search_read_complete_is_preserved(self) -> None:
        registry = build_qa_tool_registry(_Index())
        adapter = QARetrievalReactExecutionAdapter(
            gateway=SimpleNamespace(generate=lambda request: None),
            tool_registry=registry,
            max_turns=7,
            max_tool_calls=4,
            task_type=None,
            completion_policy="required_evidence",
            retrieval_tool_id=TRIVIAQA_QA_MEMORY_TOOL_ID,
        )
        request = self._request()
        initial_actions, initial_completion = (
            adapter._state_conditioned_action_domain(request, [])
        )
        self.assertEqual(
            frozenset({(TRIVIAQA_QA_MEMORY_TOOL_ID, "search")}),
            initial_actions,
        )
        self.assertFalse(initial_completion)

        search_observation = {
            "observation_status": "success",
            "executed_action": {
                "kind": "tool",
                "name": "search",
                "resource_id": TRIVIAQA_QA_MEMORY_TOOL_ID,
                "skill_id": None,
                "arguments": {"query": "author novel", "limit": 3},
            },
            "result": {
                "operation": "search",
                "query": "author novel",
                "top_k": 3,
                "memory_ids": ["memory-001"],
                "passage_ids": ["memory-001"],
                "hits": [
                    {
                        "passage_id": "memory-001",
                        "memory_id": "memory-001",
                        "document_id": "triviaqa:tc_129",
                        "title": "Which author wrote the novel?",
                        "snippet": "Question and train-memory answer statement.",
                        "rank": 1,
                    }
                ],
            },
        }
        read_actions, read_completion = adapter._state_conditioned_action_domain(
            request,
            [search_observation],
        )
        self.assertEqual(
            frozenset({(TRIVIAQA_QA_MEMORY_TOOL_ID, "read")}),
            read_actions,
        )
        self.assertFalse(read_completion)
        read_schema = adapter._state_conditioned_response_schema(
            request,
            [search_observation],
        )
        self.assertIsNotNone(read_schema)
        assert read_schema is not None
        arguments_schema = read_schema["properties"]["arguments"]
        self.assertEqual(["memory_id"], arguments_schema["required"])
        self.assertEqual(
            ["memory-001"],
            arguments_schema["properties"]["memory_id"]["enum"],
        )
        self.assertNotIn("passage_id", arguments_schema["properties"])

        read_observation = {
            "observation_status": "success",
            "executed_action": {
                "kind": "tool",
                "name": "read",
                "resource_id": TRIVIAQA_QA_MEMORY_TOOL_ID,
                "skill_id": None,
                "arguments": {"memory_id": "memory-001"},
            },
            "result": {
                "operation": "read",
                "memory_id": "memory-001",
                "passage_id": "memory-001",
                "memory": {
                    "memory_id": "memory-001",
                    "passage_id": "memory-001",
                    "title": "Which author wrote the novel?",
                    "text": "The answer is Ada.",
                },
                "passage": {
                    "memory_id": "memory-001",
                    "passage_id": "memory-001",
                    "title": "Which author wrote the novel?",
                    "text": "The answer is Ada.",
                },
            },
        }
        final_actions, completion = adapter._state_conditioned_action_domain(
            request,
            [search_observation, read_observation],
        )
        self.assertEqual(frozenset(), final_actions)
        self.assertTrue(completion)

        contract = adapter._contract(request, [search_observation])
        self.assertIn("resource_id triviaqa.qa_memory", contract)
        self.assertIn("exactly memory_id", contract)
        self.assertIn("structured semantic artifact", contract)
        self.assertNotIn("qa-retrieval", contract)
        self.assertNotIn("arguments object contains only passage_id", contract)

    def test_semantic_completion_binds_provenance_to_successful_memory_id(
        self,
    ) -> None:
        adapter = QARetrievalReactExecutionAdapter(
            gateway=SimpleNamespace(generate=lambda request: None),
            tool_registry=build_qa_tool_registry(_Index()),
            max_turns=7,
            max_tool_calls=4,
            task_type="factual_qa",
            completion_policy="required_evidence",
            retrieval_tool_id=TRIVIAQA_QA_MEMORY_TOOL_ID,
        )
        request = replace(
            self._request(),
            semantic_protocol=QA_VERIFIED_ANSWER_LINEAGE_PROTOCOL,
        )
        search_observation = {
            "observation_status": "success",
            "executed_action": {
                "kind": "tool",
                "name": "search",
                "resource_id": TRIVIAQA_QA_MEMORY_TOOL_ID,
                "skill_id": None,
                "arguments": {"query": "author novel", "limit": 3},
            },
            "result": {
                "operation": "search",
                "query": "author novel",
                "top_k": 3,
                "memory_ids": ["memory-001"],
                "passage_ids": ["memory-001"],
                "hits": [
                    {
                        "passage_id": "memory-001",
                        "memory_id": "memory-001",
                        "document_id": "triviaqa:tc_129",
                        "title": "Which author wrote the novel?",
                        "snippet": "The answer is Ada",
                        "rank": 1,
                    }
                ],
            },
        }
        read_observation = {
            "observation_status": "success",
            "executed_action": {
                "kind": "tool",
                "name": "read",
                "resource_id": TRIVIAQA_QA_MEMORY_TOOL_ID,
                "skill_id": None,
                "arguments": {"memory_id": "memory-001"},
            },
            "result": {
                "operation": "read",
                "memory_id": "memory-001",
                "passage_id": "memory-001",
                "memory": {
                    "memory_id": "memory-001",
                    "passage_id": "memory-001",
                    "title": "Which author wrote the novel?",
                    "text": "Question: Which author wrote the novel?\n"
                    "Answer: The answer is Ada",
                },
                "passage": {
                    "memory_id": "memory-001",
                    "passage_id": "memory-001",
                    "title": "Which author wrote the novel?",
                    "text": "Question: Which author wrote the novel?\n"
                    "Answer: The answer is Ada",
                },
            },
        }

        schema = adapter._state_conditioned_response_schema(
            request,
            [search_observation, read_observation],
        )
        self.assertIsNotNone(schema)
        assert schema is not None
        passage_schema = schema["properties"]["arguments"]["properties"][
            "value"
        ]["properties"]["passage_id"]
        self.assertEqual(["memory-001"], passage_schema["enum"])
        self.assertIn("successful triviaqa.qa_memory read", passage_schema["description"])

        contract = adapter._contract(
            request,
            [search_observation, read_observation],
        )
        self.assertIn("read(memory_id)", contract)
        self.assertIn("never copy source_train_task_id", contract)

    def test_canonical_memory_read_passes_strict_semantic_provenance(self) -> None:
        question = "Which author wrote the novel?"
        memory_id = "memory-001"
        receipt = {
            "tool_id": TRIVIAQA_QA_MEMORY_TOOL_ID,
            "tool_version": "qa-memory-index-v1",
            "request": {
                "action": "read",
                "arguments": {"memory_id": memory_id},
            },
            "result": {
                "completed": True,
                "value": {
                    "operation": "read",
                    "memory_id": memory_id,
                    "memory": {
                        "memory_id": memory_id,
                        "title": question,
                        "text": "Ada wrote the novel.",
                    },
                },
            },
            "error_type": None,
        }
        artifact = {
            "question_scope": question,
            "entity_identity": {
                "question_surface": "the novel",
                "evidence_surface": "the novel",
            },
            "target_relation": "wrote",
            "answer_type_constraint": "entity",
            "evidence_proposition": {
                "subject": "Ada",
                "predicate": "wrote",
                "object_or_attribute_value": "the novel",
            },
            "evidence_span": "Ada wrote the novel.",
            "passage_id": memory_id,
        }

        self.assertTrue(
            AgentWorkflowEnv._successful_read_receipt(
                receipt,
                TRIVIAQA_QA_MEMORY_TOOL_ID,
            )
        )
        self.assertEqual(
            "Ada wrote the novel.",
            AgentWorkflowEnv._successful_read_text(
                receipt,
                TRIVIAQA_QA_MEMORY_TOOL_ID,
            ),
        )
        self.assertIsNone(
            QARetrievalReactExecutionAdapter._evidence_retriever_completion_issue(
                original_question=question,
                artifact=json.dumps(artifact),
                tool_receipts=(receipt,),
                retrieval_tool_id=TRIVIAQA_QA_MEMORY_TOOL_ID,
            )
        )

        artifact["passage_id"] = "triviaqa:tc_129"
        self.assertIn(
            "no matching successful",
            QARetrievalReactExecutionAdapter._evidence_retriever_completion_issue(
                original_question=question,
                artifact=json.dumps(artifact),
                tool_receipts=(receipt,),
                retrieval_tool_id=TRIVIAQA_QA_MEMORY_TOOL_ID,
            ),
        )


if __name__ == "__main__":
    unittest.main()
