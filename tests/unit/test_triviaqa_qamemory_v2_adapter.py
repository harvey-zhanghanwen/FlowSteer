from __future__ import annotations

from dataclasses import dataclass, replace
import json
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch

from src.interactive.agent_graph import AgentNode
from src.interactive.agent_workflow_env import AgentWorkflowEnv
from src.interactive.agent_runtime import (
    AgentRequest,
    AgentResponse,
    ExecutionPhase,
    UpstreamMessage,
)
from src.interactive.model_registry import ModelSpec, ProviderSpec
from src.interactive.qa_tool_adapter import (
    QA_VERIFIED_ANSWER_LINEAGE_PROTOCOL,
    QARetrievalReactExecutionAdapter,
    TRIVIAQA_QA_MEMORY_TOOL_ID,
    _normalized_retrieval_query,
    _question_entity_anchor_tokens,
    build_qa_tool_registry,
)
from src.interactive.react_execution import (
    ReactExecutionError,
    ToolReactExecutionAdapter,
)
from src.interactive.scientific_sampling import (
    ScientificSamplingCoordinate,
    scientific_sampling_schedule_hash,
    stable_hash,
)
from src.interactive.tool_runtime import ActionKind, StructuredAction, ToolRequest


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

    def test_open_relation_query_rejects_entity_only_search_before_dispatch(
        self,
    ) -> None:
        question = (
            "At which university did Alice Example earn a doctorate in "
            "philosophy?"
        )
        request = replace(
            self._request(),
            problem=question,
            semantic_protocol=QA_VERIFIED_ANSWER_LINEAGE_PROTOCOL,
        )
        adapter = QARetrievalReactExecutionAdapter(
            gateway=SimpleNamespace(generate=lambda request: None),
            tool_registry=build_qa_tool_registry(_Index()),
            max_turns=16,
            max_tool_calls=12,
            task_type="factual_qa",
            completion_policy="required_evidence",
            retrieval_tool_id=TRIVIAQA_QA_MEMORY_TOOL_ID,
        )

        issue = adapter._tool_action_error(
            request=request,
            action=StructuredAction(
                ActionKind.TOOL,
                "search",
                {"query": "Alice Example", "limit": 3},
                resource_id=TRIVIAQA_QA_MEMORY_TOOL_ID,
            ),
            observations=[],
        )
        self.assertIsNotNone(issue)
        assert issue is not None
        self.assertTrue(
            issue.startswith("qa_retrieval_query_target_relation_loss")
        )
        self.assertIn("missing_relation_context_tokens", issue)
        self.assertIsNone(
            adapter._tool_action_error(
                request=request,
                action=StructuredAction(
                    ActionKind.TOOL,
                    "search",
                    {
                        "query": (
                            "Alice Example university doctorate philosophy"
                        ),
                        "limit": 3,
                    },
                    resource_id=TRIVIAQA_QA_MEMORY_TOOL_ID,
                ),
                observations=[],
            )
        )

    def test_malformed_possessive_is_normalized_before_query_admission(
        self,
    ) -> None:
        question = "What was John Glenn/'s first spacecraft called?"
        request = replace(
            self._request(),
            problem=question,
            semantic_protocol=QA_VERIFIED_ANSWER_LINEAGE_PROTOCOL,
        )
        adapter = QARetrievalReactExecutionAdapter(
            gateway=SimpleNamespace(generate=lambda request: None),
            tool_registry=build_qa_tool_registry(_Index()),
            max_turns=16,
            max_tool_calls=12,
            task_type="factual_qa",
            completion_policy="required_evidence",
            retrieval_tool_id=TRIVIAQA_QA_MEMORY_TOOL_ID,
        )

        self.assertEqual(
            _normalized_retrieval_query("John Glenn's first spacecraft called"),
            _normalized_retrieval_query("John Glenn/'s first spacecraft called"),
        )
        self.assertEqual(
            ("john", "glenn's"),
            _question_entity_anchor_tokens(question),
        )
        self.assertIsNone(
            adapter._tool_action_error(
                request=request,
                action=StructuredAction(
                    ActionKind.TOOL,
                    "search",
                    {
                        "query": "John Glenn first spacecraft called",
                        "limit": 3,
                    },
                    resource_id=TRIVIAQA_QA_MEMORY_TOOL_ID,
                ),
                observations=[],
            )
        )

    async def test_qamemory_failure_keeps_the_public_query_task_id(self) -> None:
        task_id = "triviaqa:tc_public"
        coordinate = ScientificSamplingCoordinate(
            sampling_schedule_hash=scientific_sampling_schedule_hash(
                base_seed=17
            ),
            schedule_purpose="triviaqa-qamemory-test",
            ordered_sequence_hash=stable_hash([task_id]),
            sequence_position=0,
            task_id=task_id,
            optimizer_step_or_anchor_ordinal=0,
        )
        adapter = QARetrievalReactExecutionAdapter(
            gateway=SimpleNamespace(generate=lambda request: None),
            tool_registry=build_qa_tool_registry(_Index()),
            max_turns=16,
            max_tool_calls=12,
            task_type="factual_qa",
            completion_policy="required_evidence",
            retrieval_tool_id=TRIVIAQA_QA_MEMORY_TOOL_ID,
            sampling_base_seed=17,
            sampling_coordinate=coordinate,
        )
        request = replace(
            self._request(),
            problem="Which author wrote the novel?",
            semantic_protocol=QA_VERIFIED_ANSWER_LINEAGE_PROTOCOL,
        )
        failure = ReactExecutionError("bounded retrieval continuation")

        with patch.object(
            ToolReactExecutionAdapter,
            "execute",
            new=AsyncMock(side_effect=failure),
        ):
            with self.assertRaises(ReactExecutionError) as caught:
                await adapter.execute(request)

        self.assertEqual(
            task_id,
            caught.exception.qa_memory_query_task_id,
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
        search_wire = json.dumps(search_result.value, ensure_ascii=False)
        self.assertNotIn("canonical_answer", search_wire)
        self.assertNotIn("paraphrase_answer_statement", search_wire)
        self.assertNotIn("Ada", search_wire)

        read_result, read_receipt = await registry.ainvoke_with_receipt(
            TRIVIAQA_QA_MEMORY_TOOL_ID,
            ToolRequest("read", {"memory_id": "memory-001"}),
        )
        self.assertIsNotNone(read_result)
        assert read_result is not None
        self.assertEqual("memory-001", read_result.value["memory_id"])
        self.assertEqual("memory-001", read_result.value["passage_id"])
        self.assertEqual(read_result.value["memory"], read_result.value["passage"])
        self.assertEqual(
            "The answer is Ada",
            read_result.value["memory"]["paraphrase_answer_statement"],
        )
        self.assertEqual("Ada", read_result.value["memory"]["canonical_answer"])
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

    async def test_semantic_react_execution_routes_complete_top_k_batch(self) -> None:
        rows = (
            ("memory-001", "triviaqa:tc_129", "Ada", 0.93),
            ("memory-002", "triviaqa:tc_130", "Grace", 0.88),
            ("memory-003", "triviaqa:tc_131", "Charles", 0.81),
        )

        class _TopKIndex:
            manifest = _Manifest()

            def __init__(self) -> None:
                self.read_calls: list[str] = []

            def search(self, query: str, *, limit: int) -> tuple[object, ...]:
                self.assert_search(query, limit)
                return tuple(
                    SimpleNamespace(
                        passage_id=memory_id,
                        memory_id=memory_id,
                        document_id=source_id,
                        source_train_task_id=source_id,
                        base_task_id=source_id,
                        cycled_training_sample=False,
                        cycle_index=None,
                        paraphrase_question=f"Candidate question {rank}",
                        paraphrase_answer_statement=f"The answer is {answer}",
                        canonical_answer=answer,
                        paraphrase_version="semantic-preserving-v1",
                        paraphrase_provenance=None,
                        title=f"Candidate question {rank}",
                        snippet=f"Candidate question {rank}",
                        rank=rank,
                        similarity=similarity,
                    )
                    for rank, (
                        memory_id,
                        source_id,
                        answer,
                        similarity,
                    ) in enumerate(rows, start=1)
                )

            @staticmethod
            def assert_search(query: str, limit: int) -> None:
                if (query, limit) != ("author novel", 3):
                    raise AssertionError("unexpected embedding search")

            def read(self, memory_id: str) -> object:
                self.read_calls.append(memory_id)
                rank, row = next(
                    (rank, row)
                    for rank, row in enumerate(rows, start=1)
                    if row[0] == memory_id
                )
                _, source_id, answer, _ = row
                return SimpleNamespace(
                    passage_id=memory_id,
                    memory_id=memory_id,
                    document_id=source_id,
                    source_train_task_id=source_id,
                    base_task_id=source_id,
                    cycled_training_sample=False,
                    cycle_index=None,
                    paraphrase_question=f"Candidate question {rank}",
                    paraphrase_answer_statement=f"The answer is {answer}",
                    canonical_answer=answer,
                    paraphrase_version="semantic-preserving-v1",
                    paraphrase_provenance=None,
                    title=f"Candidate question {rank}",
                    text=(
                        f"Question: Candidate question {rank}\n"
                        f"Answer: The answer is {answer}"
                    ),
                )

        def action(name: str, arguments: object) -> str:
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
                    *(
                        action("read", {"memory_id": memory_id})
                        for memory_id, _, _, _ in rows
                    ),
                    action(
                        "complete",
                        {"value": {"memory_ids": [row[0] for row in rows]}},
                    ),
                ]

            async def generate(self, request: AgentRequest) -> AgentResponse:
                if request.agent.allowed_tools != (
                    TRIVIAQA_QA_MEMORY_TOOL_ID,
                ):
                    raise AssertionError("QA-memory Tool must remain worker-owned")
                return AgentResponse(self.outputs.pop(0))

        index = _TopKIndex()
        response = await QARetrievalReactExecutionAdapter(
            gateway=_Gateway(),
            tool_registry=build_qa_tool_registry(index),
            max_turns=5,
            max_tool_calls=4,
            task_type="factual_qa",
            completion_policy="required_evidence",
            retrieval_tool_id=TRIVIAQA_QA_MEMORY_TOOL_ID,
        ).execute(
            replace(
                self._request(),
                semantic_protocol=QA_VERIFIED_ANSWER_LINEAGE_PROTOCOL,
            )
        )

        artifact = json.loads(response.text)
        self.assertEqual(
            [row[0] for row in rows],
            [candidate["memory_id"] for candidate in artifact["candidates"]],
        )
        self.assertEqual(4, len(response.metadata["tool_receipts"]))
        self.assertEqual([row[0] for row in rows], index.read_calls)

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
                    "source_train_task_id": "triviaqa:tc_129",
                    "paraphrase_question": "Which author wrote the novel?",
                    "paraphrase_answer_statement": "The answer is Ada",
                    "canonical_answer": "Ada",
                    "title": "Which author wrote the novel?",
                    "text": "The answer is Ada.",
                },
                "passage": {
                    "memory_id": "memory-001",
                    "passage_id": "memory-001",
                    "source_train_task_id": "triviaqa:tc_129",
                    "paraphrase_question": "Which author wrote the novel?",
                    "paraphrase_answer_statement": "The answer is Ada",
                    "canonical_answer": "Ada",
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
        self.assertIn("contains only memory_id", contract)
        self.assertIn("structured evidence artifact", contract)
        self.assertNotIn("qa-retrieval", contract)
        self.assertNotIn("arguments object contains only passage_id", contract)

    def test_semantic_completion_reads_complete_embedding_top_k_in_rank_order(
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
        ranked_memories = (
            ("memory-001", "triviaqa:tc_129", "Ada", 0.93),
            ("memory-002", "triviaqa:tc_130", "Grace", 0.88),
            ("memory-003", "triviaqa:tc_131", "Charles", 0.81),
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
                "memory_ids": [item[0] for item in ranked_memories],
                "passage_ids": [item[0] for item in ranked_memories],
                "hits": [
                    {
                        "passage_id": memory_id,
                        "memory_id": memory_id,
                        "document_id": source_id,
                        "title": f"Candidate question {rank}",
                        "snippet": f"Candidate question {rank}",
                        "rank": rank,
                        "similarity": similarity,
                    }
                    for rank, (memory_id, source_id, _, similarity) in enumerate(
                        ranked_memories,
                        start=1,
                    )
                ],
            },
        }
        observations = [search_observation]
        for rank, (memory_id, source_id, answer, _) in enumerate(
            ranked_memories,
            start=1,
        ):
            actions, completion = adapter._state_conditioned_action_domain(
                request,
                observations,
            )
            self.assertEqual(
                frozenset({(TRIVIAQA_QA_MEMORY_TOOL_ID, "read")}),
                actions,
            )
            self.assertFalse(completion)
            if rank == 1:
                self.assertIn(
                    "read_order_mismatch",
                    adapter._tool_action_error(
                        request=request,
                        action=StructuredAction(
                            ActionKind.TOOL,
                            "read",
                            {"memory_id": "memory-002"},
                            resource_id=TRIVIAQA_QA_MEMORY_TOOL_ID,
                        ),
                        observations=observations,
                    ),
                )
            read_schema = adapter._state_conditioned_response_schema(
                request,
                observations,
            )
            self.assertIsNotNone(read_schema)
            assert read_schema is not None
            self.assertEqual(
                memory_id,
                read_schema["properties"]["arguments"]["properties"][
                    "memory_id"
                ]["const"],
            )
            memory = {
                "memory_id": memory_id,
                "passage_id": memory_id,
                "source_train_task_id": source_id,
                "paraphrase_question": f"Candidate question {rank}",
                "paraphrase_answer_statement": f"The answer is {answer}",
                "canonical_answer": answer,
                "title": f"Candidate question {rank}",
                "text": f"Question: Candidate question {rank}\nAnswer: {answer}",
            }
            observations.append(
                {
                    "observation_status": "success",
                    "executed_action": {
                        "kind": "tool",
                        "name": "read",
                        "resource_id": TRIVIAQA_QA_MEMORY_TOOL_ID,
                        "skill_id": None,
                        "arguments": {"memory_id": memory_id},
                    },
                    "result": {
                        "operation": "read",
                        "memory_id": memory_id,
                        "passage_id": memory_id,
                        "memory": memory,
                        "passage": memory,
                    },
                }
            )

        actions, completion = adapter._state_conditioned_action_domain(
            request,
            observations,
        )
        self.assertEqual(frozenset(), actions)
        self.assertTrue(completion)
        schema = adapter._state_conditioned_response_schema(request, observations)
        self.assertIsNotNone(schema)
        assert schema is not None
        value_schema = schema["properties"]["arguments"]["properties"]["value"]
        self.assertEqual(["memory_ids"], value_schema["required"])
        self.assertEqual({"memory_ids"}, set(value_schema["properties"]))
        self.assertNotIn("entity_identity", value_schema["properties"])
        self.assertNotIn("evidence_proposition", value_schema["properties"])
        self.assertEqual(
            [item[0] for item in ranked_memories],
            value_schema["properties"]["memory_ids"]["const"],
        )

        retrieval_state = adapter._required_evidence_state(
            request,
            observations,
        )
        self.assertEqual(1, len(retrieval_state.retrieval_attempts))
        self.assertEqual(3, retrieval_state.retrieval_attempts[0].required_top_k)
        self.assertEqual(3, retrieval_state.retrieval_attempts[0].observed_top_k)
        self.assertEqual(
            tuple(item[0] for item in ranked_memories),
            retrieval_state.read_passage_ids,
        )

        contract = adapter._contract(
            request,
            observations,
        )
        self.assertIn("read(memory_id)", contract)
        self.assertIn("arguments.value must contain only memory_ids", contract)
        self.assertIn("complete ordered candidates list", contract)

    def test_native_top_k_artifact_requires_ordered_search_and_all_reads(
        self,
    ) -> None:
        question = "Which author wrote the novel?"
        rows = (
            ("memory-001", "triviaqa:tc_129", "Ada", 0.93),
            ("memory-002", "triviaqa:tc_130", "Grace", 0.88),
            ("memory-003", "triviaqa:tc_131", "Charles", 0.81),
        )
        memory_ids = [row[0] for row in rows]
        search_receipt = {
            "tool_id": TRIVIAQA_QA_MEMORY_TOOL_ID,
            "tool_version": "qa-memory-index-v1",
            "request": {
                "action": "search",
                "arguments": {"query": "author novel", "limit": 3},
            },
            "result": {
                "completed": True,
                "value": {
                    "operation": "search",
                    "query": "author novel",
                    "top_k": 3,
                    "memory_ids": memory_ids,
                    "passage_ids": memory_ids,
                    "hits": [
                        {
                            "memory_id": memory_id,
                            "passage_id": memory_id,
                            "document_id": source_id,
                            "source_train_task_id": source_id,
                            "paraphrase_question": f"Candidate question {rank}",
                            "title": f"Candidate question {rank}",
                            "snippet": f"Candidate question {rank}",
                            "rank": rank,
                            "similarity": similarity,
                        }
                        for rank, (
                            memory_id,
                            source_id,
                            _,
                            similarity,
                        ) in enumerate(rows, start=1)
                    ],
                },
            },
            "error_type": None,
        }
        read_receipts = tuple(
            {
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
                            "source_train_task_id": source_id,
                            "paraphrase_question": f"Candidate question {rank}",
                            "paraphrase_answer_statement": f"The answer is {answer}",
                            "canonical_answer": answer,
                            "title": f"Candidate question {rank}",
                            "text": (
                                f"Question: Candidate question {rank}\n"
                                f"Answer: The answer is {answer}"
                            ),
                        },
                    },
                },
                "error_type": None,
            }
            for rank, (memory_id, source_id, answer, _) in enumerate(
                rows,
                start=1,
            )
        )
        receipts = (search_receipt, *read_receipts)

        self.assertTrue(
            AgentWorkflowEnv._successful_read_receipt(
                read_receipts[0],
                TRIVIAQA_QA_MEMORY_TOOL_ID,
            )
        )
        self.assertEqual(
            "Question: Candidate question 1\nAnswer: The answer is Ada",
            AgentWorkflowEnv._successful_read_text(
                read_receipts[0],
                TRIVIAQA_QA_MEMORY_TOOL_ID,
            ),
        )
        projected, projection_issue = (
            QARetrievalReactExecutionAdapter._qa_memory_completion_receipt_projection(
                original_question=question,
                selection_artifact=json.dumps({"memory_ids": memory_ids}),
                tool_receipts=receipts,
            )
        )
        self.assertIsNone(projection_issue)
        self.assertIsNotNone(projected)
        assert projected is not None
        artifact = json.loads(projected)
        self.assertEqual(question, artifact["question_scope"])
        self.assertEqual("author novel", artifact["retrieval_query"])
        self.assertEqual(3, artifact["top_k"])
        self.assertEqual(memory_ids, [item["memory_id"] for item in artifact["candidates"]])
        self.assertEqual([1, 2, 3], [item["rank"] for item in artifact["candidates"]])
        self.assertEqual([0.93, 0.88, 0.81], [item["similarity"] for item in artifact["candidates"]])

        exact_source_projected, exact_source_issue = (
            QARetrievalReactExecutionAdapter._qa_memory_completion_receipt_projection(
                original_question=question,
                selection_artifact=json.dumps(
                    {
                        "memory_ids": memory_ids,
                        "retrieval_status": "knowledge_base_coverage_failure",
                        "relevant_memory_ids": [],
                    }
                ),
                tool_receipts=receipts,
                parametric_fallback_after_coverage_failure=True,
                expected_source_task_id="triviaqa:tc_129",
            )
        )
        self.assertIsNone(exact_source_issue)
        self.assertIsNotNone(exact_source_projected)
        assert exact_source_projected is not None
        exact_source_artifact = json.loads(exact_source_projected)
        self.assertEqual(
            "evidence_found",
            exact_source_artifact["retrieval_status"],
        )
        self.assertEqual(
            ["memory-001"],
            exact_source_artifact["relevant_memory_ids"],
        )
        self.assertIsNone(
            QARetrievalReactExecutionAdapter._evidence_retriever_completion_issue(
                original_question=question,
                artifact=exact_source_projected,
                tool_receipts=receipts,
                retrieval_tool_id=TRIVIAQA_QA_MEMORY_TOOL_ID,
                parametric_fallback_after_coverage_failure=True,
                expected_source_task_id="triviaqa:tc_129",
            )
        )
        self.assertIsNone(
            QARetrievalReactExecutionAdapter._evidence_retriever_completion_issue(
                original_question=question,
                artifact=json.dumps(artifact),
                tool_receipts=receipts,
                retrieval_tool_id=TRIVIAQA_QA_MEMORY_TOOL_ID,
            )
        )

        adapter = QARetrievalReactExecutionAdapter(
            gateway=SimpleNamespace(generate=lambda request: None),
            tool_registry=build_qa_tool_registry(_Index()),
            max_turns=7,
            max_tool_calls=4,
            task_type="factual_qa",
            completion_policy="required_evidence",
            retrieval_tool_id=TRIVIAQA_QA_MEMORY_TOOL_ID,
        )
        reasoner_request = replace(
            self._request(),
            agent=AgentNode(
                "reasoner",
                "model",
                "reason from evidence",
                role_family="reasoner",
            ),
            semantic_protocol=QA_VERIFIED_ANSWER_LINEAGE_PROTOCOL,
            upstream=(
                UpstreamMessage(
                    source_agent_id="retriever",
                    target_agent_id="reasoner",
                    content=json.dumps(artifact),
                    tool_receipts=receipts,
                ),
            ),
        )
        self.assertEqual(
            receipts,
            adapter._validated_upstream_evidence_receipts(reasoner_request),
        )

        incomplete, incomplete_issue = (
            QARetrievalReactExecutionAdapter._qa_memory_completion_receipt_projection(
                original_question=question,
                selection_artifact=json.dumps({"memory_ids": memory_ids}),
                tool_receipts=(search_receipt, *read_receipts[:2]),
            )
        )
        self.assertIsNone(incomplete)
        self.assertIn("every embedding hit", incomplete_issue)

        out_of_order, order_issue = (
            QARetrievalReactExecutionAdapter._qa_memory_completion_receipt_projection(
                original_question=question,
                selection_artifact=json.dumps({"memory_ids": memory_ids}),
                tool_receipts=(
                    search_receipt,
                    read_receipts[1],
                    read_receipts[0],
                    read_receipts[2],
                ),
            )
        )
        self.assertIsNone(out_of_order)
        self.assertIn("original rank order", order_issue)

        mismatched = json.loads(json.dumps(artifact))
        mismatched["candidates"][1]["canonical_answer"] = "Wrong"
        self.assertIn(
            "field 'candidates'",
            QARetrievalReactExecutionAdapter._evidence_retriever_completion_issue(
                original_question=question,
                artifact=json.dumps(mismatched),
                tool_receipts=receipts,
                retrieval_tool_id=TRIVIAQA_QA_MEMORY_TOOL_ID,
            ),
        )


if __name__ == "__main__":
    unittest.main()
