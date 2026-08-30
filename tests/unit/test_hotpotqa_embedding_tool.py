from __future__ import annotations

import asyncio
from dataclasses import dataclass
import json
from typing import Any
import unittest

from src.interactive.agent_graph import AgentGraph, AgentNode, AgentRelation
from src.interactive.agent_runtime import (
    AgentRuntime,
    AgentRequest,
    AgentResponse,
    ExecutionPhase,
)
from src.interactive.hotpotqa_embedding_tool import (
    HOTPOTQA_QA_MEMORY_TOOL_ID,
    HOTPOTQA_RETRIEVAL_TOOL_ID,
    HotpotQAEmbeddingReactExecutionAdapter,
    build_hotpotqa_embedding_tool_registry,
)
from src.interactive.agent_workflow_env import AgentWorkflowEnv
from src.interactive.director import AgentGraphOrchestrator
from src.interactive.model_registry import (
    ModelRegistry,
    ModelSpec,
    ProviderSpec,
)
from src.interactive.rollout_collector import SGLangReceiptDirectorClient
from src.interactive.tool_runtime import ToolRequest


@dataclass(frozen=True)
class _Manifest:
    index_id: str = "hotpot-public-embedding-v1"
    corpus_version: str = "hotpotqa-public-context-v1"
    source: str = "hotpot_qa"
    split: str = "validation"
    embedding_model: str = "test-encoder"
    embedding_dimension: int = 3
    normalized: bool = True
    similarity: str = "cosine"
    frozen_top_k: int = 2
    document_count: int = 2
    passage_count: int = 2


@dataclass(frozen=True)
class _Hit:
    passage_id: str
    document_id: str
    title: str
    snippet: str
    similarity: float
    rank: int
    # These fields simulate evaluator-private data accidentally present on a
    # backend object.  The adapter must never project them into observations.
    answer: str = "PRIVATE ANSWER"
    supporting_facts: tuple[str, ...] = ("PRIVATE SUPPORT",)


@dataclass(frozen=True)
class _Passage:
    passage_id: str
    document_id: str
    title: str
    text: str
    answer: str = "PRIVATE ANSWER"
    evaluator_receipt: str = "PRIVATE RECEIPT"


class _Index:
    manifest = _Manifest()

    def __init__(self) -> None:
        self.search_calls: list[tuple[str, str, int]] = []
        self.read_calls: list[tuple[str, str]] = []

    def search(self, task_id: str, query: str, k: int) -> tuple[_Hit, ...]:
        self.search_calls.append((task_id, query, k))
        return (
            _Hit("p1", "d1", "Alpha", "First public passage.", 0.91, 1),
            _Hit("p2", "d2", "Beta", "Second public passage.", 0.72, 2),
        )

    def read(self, task_id: str, doc_id: str) -> _Passage:
        self.read_calls.append((task_id, doc_id))
        values = {
            "p1": _Passage("p1", "d1", "Alpha", "Full public passage one."),
            "p2": _Passage("p2", "d2", "Beta", "Full public passage two."),
        }
        return values[doc_id]


@dataclass(frozen=True)
class _QAMemoryManifest:
    schema_version: str = "flowsteer.hotpotqa.qa_memory_index.v1"
    index_id: str = "hotpotqa-train-qa-memory-test-v1"
    corpus_version: str = "flowsteer.hotpotqa.train_qa_memory.v1"
    source: str = "HotpotQA aligned frozen train"
    source_split: str = "train"
    embedding_model: str = "test-encoder"
    embedding_dimension: int = 3
    normalized: bool = True
    similarity: str = "cosine"
    frozen_top_k: int = 2
    train_record_count: int = 512
    unique_source_count: int = 400
    cycled_record_count: int = 112
    paraphrase_count: int = 512
    heldout_validation_count: int = 128
    validation_overlap_count: int = 0
    paraphrase_versions: tuple[str, ...] = ("semantic-paraphrase-v1",)
    paraphrase_provenances: tuple[str, ...] = ("offline-train-only",)


@dataclass(frozen=True)
class _QAMemoryHit:
    memory_id: str
    source_train_task_id: str
    paraphrase_question: str
    paraphrase_answer_statement: str
    similarity: float
    rank: int
    evaluator_receipt: str = "PRIVATE RECEIPT"


@dataclass(frozen=True)
class _QAMemory:
    memory_id: str
    source_train_task_id: str
    base_task_id: str
    cycled: bool
    paraphrase_question: str
    paraphrase_answer_statement: str
    canonical_answer: str
    paraphrase_version: str
    paraphrase_provenance: str
    supporting_facts: tuple[str, ...] = ("PRIVATE SUPPORT",)


class _QAMemoryIndex:
    manifest = _QAMemoryManifest()

    def __init__(self) -> None:
        self.search_calls: list[tuple[str, int]] = []
        self.read_calls: list[str] = []

    def search(self, query: str, k: int) -> tuple[_QAMemoryHit, ...]:
        self.search_calls.append((query, k))
        return (
            _QAMemoryHit(
                "memory-1",
                "hotpotqa:train-a",
                "Which person wrote Alpha?",
                "The writer of Alpha is Ada Lovelace.",
                0.93,
                1,
            ),
            _QAMemoryHit(
                "memory-2",
                "hotpotqa:train-b",
                "Who authored Beta?",
                "Beta was authored by Grace Hopper.",
                0.81,
                2,
            ),
        )

    def read(self, memory_id: str) -> _QAMemory:
        self.read_calls.append(memory_id)
        values = {
            "memory-1": _QAMemory(
                "memory-1",
                "hotpotqa:train-a",
                "hotpotqa:train-a",
                False,
                "Which person wrote Alpha?",
                "The writer of Alpha is Ada Lovelace.",
                "Ada Lovelace",
                "semantic-paraphrase-v1",
                "offline-train-only",
            ),
            "memory-2": _QAMemory(
                "memory-2",
                "hotpotqa:train-b",
                "hotpotqa:train-b",
                False,
                "Who authored Beta?",
                "Beta was authored by Grace Hopper.",
                "Grace Hopper",
                "semantic-paraphrase-v1",
                "offline-train-only",
            ),
        }
        return values[memory_id]


def _action(
    kind: str,
    *,
    name: str,
    arguments: object,
    resource_id: str | None,
) -> str:
    return json.dumps(
        {
            "kind": kind,
            "name": name,
            "arguments": arguments,
            "resource_id": resource_id,
            "skill_id": None,
        }
    )


class _SequenceGateway:
    def __init__(self, outputs: list[str]) -> None:
        self.outputs = list(outputs)
        self.requests: list[AgentRequest] = []

    async def generate(self, request: AgentRequest) -> AgentResponse:
        self.requests.append(request)
        return AgentResponse(self.outputs.pop(0), {})


class _UnusedDirectorClient:
    async def propose(self, prompt: str, *, seed: int | None = None):
        raise AssertionError("the no-model dataflow test does not call the Director")


class _DirectorTokenizer:
    def apply_chat_template(self, messages: object, **kwargs: object) -> list[int]:
        return [101, 102]

    def decode(self, token_ids: object, **kwargs: object) -> str:
        return ""


def _react_request() -> AgentRequest:
    return AgentRequest(
        request_id="run:1:retriever:single",
        run_id="run",
        graph_revision=1,
        problem="Which of Alpha and Beta has the larger value?",
        agent=AgentNode(
            "retriever",
            "m",
            "Retrieve evidence for both Alpha and Beta, then compare them.",
            allowed_tools=(HOTPOTQA_RETRIEVAL_TOOL_ID,),
            execution_mode="react",
            artifact_type="text",
            completion_condition="Both entities have read evidence and are compared.",
        ),
        model=ModelSpec("m", "fake"),
        provider=ProviderSpec("fake", kind="test"),
        phase=ExecutionPhase.SINGLE,
    )


def _qa_memory_worker_request() -> AgentRequest:
    return AgentRequest(
        request_id="run:1:qa-memory-worker:single",
        run_id="run",
        graph_revision=1,
        problem="Who wrote Alpha?",
        agent=AgentNode(
            "qa-memory-worker",
            "m",
            "Retrieve relevant train QA-memory evidence for the question.",
            allowed_tools=(HOTPOTQA_QA_MEMORY_TOOL_ID,),
            execution_mode="react",
            artifact_type="text",
            completion_condition="Return an evidence artifact after dynamic retrieval.",
        ),
        model=ModelSpec("m", "fake"),
        provider=ProviderSpec("fake", kind="test"),
        phase=ExecutionPhase.SINGLE,
    )


def _contains_forbidden_key(value: Any) -> bool:
    forbidden = {
        "answer",
        "answers",
        "supporting_facts",
        "supporting-facts",
        "evaluator",
        "evaluator_receipt",
        "reference_answer",
    }
    if isinstance(value, dict):
        if forbidden.intersection(value):
            return True
        return any(_contains_forbidden_key(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return any(_contains_forbidden_key(item) for item in value)
    return False


class HotpotQAEmbeddingToolTests(unittest.TestCase):
    def test_public_task_reaches_director_and_worker_while_query_receipt_is_question_only(
        self,
    ) -> None:
        question_scope = "Which city is the capital of India?"
        public_task = (
            "Based on the following passages, answer the question.\n\n"
            "[[Delhi] Delhi is the capital of India.]\n\n"
            "[[Mumbai] Mumbai is a city in Maharashtra.]\n\n"
            f"Question: {question_scope}"
        )
        index = _QAMemoryIndex()
        tool_registry = build_hotpotqa_embedding_tool_registry(
            index,
            task_id="hotpotqa:public-task",
            tool_id=HOTPOTQA_QA_MEMORY_TOOL_ID,
            frozen_top_k=2,
        )
        gateway = _SequenceGateway(
            [
                _action(
                    "tool",
                    name="search",
                    arguments={"query": question_scope, "k": 2},
                    resource_id=HOTPOTQA_QA_MEMORY_TOOL_ID,
                ),
                _action(
                    "tool",
                    name="read",
                    arguments={"memory_id": "memory-1"},
                    resource_id=HOTPOTQA_QA_MEMORY_TOOL_ID,
                ),
                _action(
                    "tool",
                    name="read",
                    arguments={"memory_id": "memory-2"},
                    resource_id=HOTPOTQA_QA_MEMORY_TOOL_ID,
                ),
                _action(
                    "complete",
                    name="complete",
                    arguments={
                        "value": {
                            "retrieval_sufficiency": "unsupported",
                            "selected_memory_id": None,
                        }
                    },
                    resource_id=None,
                ),
                "<answer>Delhi</answer>",
            ]
        )
        registry = ModelRegistry(
            [ProviderSpec("fake", kind="test")],
            [ModelSpec("m", "fake")],
        )
        adapter = HotpotQAEmbeddingReactExecutionAdapter(
            gateway=gateway,
            tool_registry=tool_registry,
            retrieval_query_scope=question_scope,
            max_turns=5,
            max_tool_calls=3,
        )
        runtime = AgentRuntime(
            registry,
            gateway,
            execution_adapters={"react": adapter},
            tool_registry=tool_registry,
            dataset_id="hotpotqa",
        )
        graph = AgentGraph(
            [
                AgentNode(
                    "worker",
                    "m",
                    "Retrieve a relevant training QA-memory record.",
                    execution_mode="react",
                    allowed_tools=(HOTPOTQA_QA_MEMORY_TOOL_ID,),
                ),
                AgentNode("output", "m", "Answer from public evidence and upstream artifacts."),
            ],
            [AgentRelation("worker", "output", True, False)],
            output_agent_id="output",
        )

        env = AgentWorkflowEnv(
            registry,
            runtime=runtime,
            problem=public_task,
            graph=graph,
            execute_on_edit=True,
            director_feedback_mode="control_plane",
        )
        director = AgentGraphOrchestrator(
            registry,
            _UnusedDirectorClient(),
            tool_registry=tool_registry,
        )
        step = asyncio.run(
            env.step(
                json.dumps(
                    {
                        "action": "modify_agent",
                        "agent_id": "output",
                        "contract": (
                            "Answer from public evidence and routed upstream artifacts."
                        ),
                    }
                )
            )
        )
        self.assertTrue(step.accepted)
        self.assertIsNotNone(step.execution)
        assert step.execution is not None
        result = step.execution
        director_prompt = director.build_prompt(env, 1, ())
        director_payload = SGLangReceiptDirectorClient(
            _DirectorTokenizer(),
            policy_version="test-policy",
        ).request_payload(director_prompt, seed=1)

        self.assertIn("[[Delhi] Delhi is the capital of India.", director_prompt)
        self.assertNotIn("memory-1", director_prompt)
        self.assertNotIn("Ada Lovelace", director_prompt)
        self.assertNotIn("paraphrase_answer_statement", env.snapshot().last_feedback)
        self.assertFalse(
            {"tools", "tool_choice", "allowed_tools", "retrieval", "documents"}
            & set(director_payload)
        )
        self.assertEqual([(question_scope, 2)], index.search_calls)
        self.assertTrue(all(request.problem == public_task for request in gateway.requests))
        self.assertIn(question_scope, gateway.requests[0].agent.contract)
        self.assertIn("[[Delhi]", gateway.requests[0].problem)
        worker_call = next(
            call for call in result.calls if call.request.agent.id == "worker"
        )
        output_call = next(
            call for call in result.calls if call.request.agent.id == "output"
        )
        self.assertEqual(
            ["search", "read", "read"],
            [
                receipt["request"]["action"]
                for receipt in worker_call.response.metadata["tool_receipts"]
            ],
        )
        self.assertEqual("worker", output_call.request.upstream[0].source_agent_id)
        self.assertEqual("output", output_call.request.upstream[0].target_agent_id)
        self.assertEqual(
            3,
            len(output_call.request.upstream[0].tool_receipts),
        )

    def test_react_completes_after_one_search_read_pair(self) -> None:
        index = _Index()
        gateway = _SequenceGateway(
            [
                _action(
                    "tool",
                    name="search",
                    arguments={"query": "Alpha value", "k": 2},
                    resource_id=HOTPOTQA_RETRIEVAL_TOOL_ID,
                ),
                _action(
                    "tool",
                    name="read",
                    arguments={"doc_id": "p1"},
                    resource_id=HOTPOTQA_RETRIEVAL_TOOL_ID,
                ),
                _action(
                    "complete",
                    name="complete",
                    arguments={"value": "aligned artifact"},
                    resource_id=None,
                ),
            ]
        )
        adapter = HotpotQAEmbeddingReactExecutionAdapter(
            gateway=gateway,
            tool_registry=build_hotpotqa_embedding_tool_registry(
                index,
                task_id="task-1",
            ),
            max_turns=6,
            max_tool_calls=4,
        )

        response = asyncio.run(adapter.execute(_react_request()))

        self.assertEqual("aligned artifact", response.text)
        self.assertEqual(2, response.metadata["tool_calls"])
        self.assertEqual(
            [("task-1", "Alpha value", 2)],
            index.search_calls,
        )
        self.assertEqual(
            [("task-1", "p1")],
            index.read_calls,
        )
        contracts = [request.agent.contract for request in gateway.requests]
        self.assertIn("search only", contracts[0])
        self.assertIn("read only", contracts[1])
        self.assertIn("complete only", contracts[2])
        response_schemas = [
            json.loads(request.model.metadata["response_json_schema"])
            for request in gateway.requests
        ]
        self.assertEqual(
            ["search", "read", "complete"],
            [schema["properties"]["name"]["const"] for schema in response_schemas],
        )
        self.assertTrue(
            all(
                set(schema["required"])
                == {"arguments", "kind", "name", "resource_id", "skill_id"}
                and schema["additionalProperties"] is False
                for schema in response_schemas
            )
        )
        self.assertEqual(
            2,
            response_schemas[0]["properties"]["arguments"]["properties"]["k"][
                "const"
            ],
        )
        self.assertEqual(
            "string",
            response_schemas[1]["properties"]["arguments"]["properties"][
                "doc_id"
            ]["type"],
        )
        self.assertEqual(
            ["value"],
            response_schemas[2]["properties"]["arguments"]["required"],
        )
        self.assertIsNone(
            response_schemas[2]["properties"]["resource_id"]["const"]
        )

    def test_action_domain_admits_completion_after_first_public_read(self) -> None:
        index = _Index()
        gateway = _SequenceGateway(
            [
                _action(
                    "tool",
                    name="search",
                    arguments={"query": "Alpha value", "k": 2},
                    resource_id=HOTPOTQA_RETRIEVAL_TOOL_ID,
                ),
                _action(
                    "tool",
                    name="read",
                    arguments={"doc_id": "p1"},
                    resource_id=HOTPOTQA_RETRIEVAL_TOOL_ID,
                ),
                _action(
                    "complete",
                    name="complete",
                    arguments={"value": "unsupported early answer"},
                    resource_id=None,
                ),
                _action(
                    "tool",
                    name="search",
                    arguments={"query": "  ALPHA   value  ", "k": 2},
                    resource_id=HOTPOTQA_RETRIEVAL_TOOL_ID,
                ),
                _action(
                    "tool",
                    name="search",
                    arguments={"query": "Beta value", "k": 2},
                    resource_id=HOTPOTQA_RETRIEVAL_TOOL_ID,
                ),
                _action(
                    "tool",
                    name="read",
                    arguments={"doc_id": "p2"},
                    resource_id=HOTPOTQA_RETRIEVAL_TOOL_ID,
                ),
                _action(
                    "complete",
                    name="complete",
                    arguments={"value": "Beta"},
                    resource_id=None,
                ),
            ]
        )
        adapter = HotpotQAEmbeddingReactExecutionAdapter(
            gateway=gateway,
            tool_registry=build_hotpotqa_embedding_tool_registry(
                index,
                task_id="task-1",
            ),
            max_turns=7,
            max_tool_calls=4,
        )

        response = asyncio.run(adapter.execute(_react_request()))

        self.assertEqual("unsupported early answer", response.text)
        self.assertEqual(2, response.metadata["tool_calls"])
        self.assertEqual(1, len(index.search_calls))

    def test_react_completion_requires_successful_search_and_read(self) -> None:
        receipt = lambda action: {
            "tool_id": HOTPOTQA_RETRIEVAL_TOOL_ID,
            "error_type": None,
            "request": {"action": action, "arguments": {}},
            "result": {"value": {}, "completed": True},
        }
        check = HotpotQAEmbeddingReactExecutionAdapter._completion_error
        self.assertEqual(
            "hotpotqa_dynamic_search_required",
            check(
                object(),
                action=object(),
                artifact="answer",
                tool_receipts=[],
            ),
        )
        self.assertEqual(
            "hotpotqa_dynamic_read_required",
            check(
                object(),
                action=object(),
                artifact="answer",
                tool_receipts=[receipt("search")],
            ),
        )
        self.assertIsNone(
            check(
                object(),
                action=object(),
                artifact="answer",
                tool_receipts=[receipt("search"), receipt("read")],
            )
        )

    def test_registry_exposes_one_resource_with_search_and_read_actions(self) -> None:
        registry = build_hotpotqa_embedding_tool_registry(
            _Index(), task_id="task-1"
        )

        self.assertEqual((HOTPOTQA_RETRIEVAL_TOOL_ID,), registry.resource_ids)
        capability = registry.require_capability(HOTPOTQA_RETRIEVAL_TOOL_ID)
        self.assertEqual(("hotpotqa",), capability.dataset_scope)
        self.assertEqual(("read", "search"), capability.action_names)
        action_schemas = {
            schema["title"]: schema for schema in capability.input_schema["oneOf"]
        }
        self.assertEqual({"read", "search"}, set(action_schemas))
        self.assertEqual(
            action_schemas["search"],
            dict(capability.action_schemas["search"]),
        )
        self.assertEqual(
            action_schemas["read"],
            dict(capability.action_schemas["read"]),
        )
        self.assertEqual(2, action_schemas["search"]["properties"]["k"]["const"])
        self.assertNotIn("task_id", action_schemas["search"]["properties"])
        self.assertIsNone(
            capability.argument_validation_error(
                "search", {"query": "Alpha", "k": 2}
            )
        )
        self.assertIn(
            "2",
            capability.argument_validation_error(
                "search", {"query": "Alpha", "k": 1}
            )["message"],
        )
        self.assertEqual("none", capability.side_effect)

    def test_search_is_task_scoped_and_receipt_preserves_ranked_hits(self) -> None:
        index = _Index()
        registry = build_hotpotqa_embedding_tool_registry(
            index, task_id="task-1"
        )

        result, receipt = asyncio.run(
            registry.ainvoke_with_receipt(
                HOTPOTQA_RETRIEVAL_TOOL_ID,
                ToolRequest("search", {"query": "Alpha relation", "k": 2}),
            )
        )

        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual([("task-1", "Alpha relation", 2)], index.search_calls)
        self.assertEqual("Alpha relation", result.value["query"])
        self.assertEqual(["p1", "p2"], result.value["doc_ids"])
        self.assertEqual([1, 2], [hit["rank"] for hit in result.value["hits"]])
        self.assertEqual(0.91, result.value["hits"][0]["similarity"])
        self.assertEqual("hotpot-public-embedding-v1", receipt.tool_version)
        self.assertEqual(
            {"query": "Alpha relation", "k": 2},
            dict(receipt.request.arguments),
        )
        self.assertFalse(_contains_forbidden_key(receipt.to_value()))

    def test_read_requires_successful_search_and_preserves_public_passage(self) -> None:
        index = _Index()
        registry = build_hotpotqa_embedding_tool_registry(
            index, task_id="task-1"
        )

        missing, missing_receipt = asyncio.run(
            registry.ainvoke_with_receipt(
                HOTPOTQA_RETRIEVAL_TOOL_ID,
                ToolRequest("read", {"doc_id": "p1"}),
            )
        )
        self.assertIsNone(missing)
        self.assertEqual("ValueError", missing_receipt.error_type)
        self.assertEqual([], index.read_calls)

        asyncio.run(
            registry.ainvoke(
                HOTPOTQA_RETRIEVAL_TOOL_ID,
                ToolRequest("search", {"query": "Alpha", "k": 2}),
            )
        )
        result, receipt = asyncio.run(
            registry.ainvoke_with_receipt(
                HOTPOTQA_RETRIEVAL_TOOL_ID,
                ToolRequest("read", {"doc_id": "p1"}),
            )
        )

        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual([("task-1", "p1")], index.read_calls)
        self.assertEqual("Full public passage one.", result.value["passage"]["content"])
        self.assertEqual("p1", result.value["doc_id"])
        self.assertFalse(_contains_forbidden_key(receipt.to_value()))

    def test_top_k_and_argument_drift_fail_closed_before_index_call(self) -> None:
        index = _Index()
        registry = build_hotpotqa_embedding_tool_registry(
            index, task_id="task-1", frozen_top_k=2
        )

        wrong_k, wrong_k_receipt = asyncio.run(
            registry.ainvoke_with_receipt(
                HOTPOTQA_RETRIEVAL_TOOL_ID,
                ToolRequest("search", {"query": "Alpha", "k": 1}),
            )
        )
        leaked, leaked_receipt = asyncio.run(
            registry.ainvoke_with_receipt(
                HOTPOTQA_RETRIEVAL_TOOL_ID,
                ToolRequest(
                    "search",
                    {
                        "query": "Alpha",
                        "k": 2,
                        "reference_answer": "PRIVATE",
                    },
                ),
            )
        )

        self.assertIsNone(wrong_k)
        self.assertEqual("ValueError", wrong_k_receipt.error_type)
        self.assertIsNone(leaked)
        self.assertEqual("ValueError", leaked_receipt.error_type)
        self.assertEqual([], index.search_calls)

    def test_qa_memory_registry_uses_global_search_and_public_memory_allowlist(
        self,
    ) -> None:
        index = _QAMemoryIndex()
        registry = build_hotpotqa_embedding_tool_registry(
            index,
            task_id="hotpotqa:heldout-validation-001",
            tool_id=HOTPOTQA_QA_MEMORY_TOOL_ID,
        )

        search_result, search_receipt = asyncio.run(
            registry.ainvoke_with_receipt(
                HOTPOTQA_QA_MEMORY_TOOL_ID,
                ToolRequest("search", {"query": "Alpha author", "k": 2}),
            )
        )

        self.assertIsNotNone(search_result)
        assert search_result is not None
        self.assertEqual([("Alpha author", 2)], index.search_calls)
        self.assertEqual([], index.read_calls)
        self.assertEqual(
            "hotpotqa:heldout-validation-001",
            search_result.value["task_id"],
        )
        self.assertEqual(
            "train_qa_memory",
            search_result.value["retrieval_index"]["corpus_kind"],
        )
        self.assertEqual(
            ["memory-1", "memory-2"], search_result.value["memory_ids"]
        )
        self.assertEqual(
            {
                "memory_id",
                "source_train_task_id",
                "paraphrase_question",
                "paraphrase_answer_statement",
                "similarity",
                "rank",
            },
            set(search_result.value["hits"][0]),
        )
        self.assertEqual(HOTPOTQA_QA_MEMORY_TOOL_ID, search_receipt.tool_id)

        read_result, read_receipt = asyncio.run(
            registry.ainvoke_with_receipt(
                HOTPOTQA_QA_MEMORY_TOOL_ID,
                ToolRequest("read", {"memory_id": "memory-1"}),
            )
        )

        self.assertIsNotNone(read_result)
        assert read_result is not None
        self.assertEqual(["memory-1"], index.read_calls)
        self.assertEqual("memory-1", read_result.value["memory_id"])
        self.assertEqual(
            {
                "memory_id",
                "source_train_task_id",
                "base_task_id",
                "cycled",
                "paraphrase_question",
                "paraphrase_answer_statement",
                "canonical_answer",
                "paraphrase_version",
                "paraphrase_provenance",
            },
            set(read_result.value["memory"]),
        )
        self.assertEqual("Ada Lovelace", read_result.value["memory"]["canonical_answer"])
        self.assertFalse(_contains_forbidden_key(search_receipt.to_value()))
        self.assertFalse(_contains_forbidden_key(read_receipt.to_value()))
        capability = registry.require_capability(HOTPOTQA_QA_MEMORY_TOOL_ID)
        self.assertEqual(
            ["memory_id"], capability.action_schemas["read"]["required"]
        )
        self.assertEqual((HOTPOTQA_QA_MEMORY_TOOL_ID,), registry.resource_ids)
        self.assertFalse(
            any("web" in resource_id.casefold() for resource_id in registry.resource_ids)
        )

    def test_qa_memory_react_receipts_are_emitted_inside_worker_execution(
        self,
    ) -> None:
        index = _QAMemoryIndex()
        gateway = _SequenceGateway(
            [
                _action(
                    "tool",
                    name="search",
                    arguments={"query": "Alpha author", "k": 2},
                    resource_id=HOTPOTQA_QA_MEMORY_TOOL_ID,
                ),
                _action(
                    "tool",
                    name="read",
                    arguments={"memory_id": "memory-1"},
                    resource_id=HOTPOTQA_QA_MEMORY_TOOL_ID,
                ),
                _action(
                    "tool",
                    name="read",
                    arguments={"memory_id": "memory-2"},
                    resource_id=HOTPOTQA_QA_MEMORY_TOOL_ID,
                ),
                _action(
                    "complete",
                    name="complete",
                    arguments={
                        "value": {
                            "retrieval_sufficiency": "supported",
                            "selected_memory_id": "memory-1",
                        }
                    },
                    resource_id=None,
                ),
            ]
        )
        adapter = HotpotQAEmbeddingReactExecutionAdapter(
            gateway=gateway,
            tool_registry=build_hotpotqa_embedding_tool_registry(
                index,
                task_id="hotpotqa:heldout-validation-001",
                tool_id=HOTPOTQA_QA_MEMORY_TOOL_ID,
            ),
            max_turns=4,
            max_tool_calls=3,
        )

        response = asyncio.run(adapter.execute(_qa_memory_worker_request()))

        self.assertEqual(
            {
                "retrieval_sufficiency": "supported",
                "selected_memory_id": "memory-1",
            },
            json.loads(response.text),
        )
        self.assertEqual([("Alpha author", 2)], index.search_calls)
        self.assertEqual(["memory-1", "memory-2"], index.read_calls)
        self.assertEqual(
            ["search", "read", "read", "complete"],
            [
                json.loads(request.model.metadata["response_json_schema"])[
                    "properties"
                ]["name"]["const"]
                for request in gateway.requests
            ],
        )
        response_schemas = [
            json.loads(request.model.metadata["response_json_schema"])
            for request in gateway.requests
        ]
        self.assertEqual(
            ["memory-1", "memory-2"],
            response_schemas[1]["properties"]["arguments"]["properties"][
                "memory_id"
            ]["enum"],
        )
        self.assertEqual(
            ["memory-2"],
            response_schemas[2]["properties"]["arguments"]["properties"][
                "memory_id"
            ]["enum"],
        )
        self.assertNotIn(
            "enum",
            adapter._tool_registry.require_capability(
                HOTPOTQA_QA_MEMORY_TOOL_ID
            ).action_schemas["read"]["properties"]["memory_id"],
        )
        completion_schema = json.loads(
            gateway.requests[-1].model.metadata["response_json_schema"]
        )["properties"]["arguments"]["properties"]["value"]
        self.assertEqual(
            ["retrieval_sufficiency", "selected_memory_id"],
            completion_schema["required"],
        )
        self.assertIn(
            "current query and public task passages",
            gateway.requests[-1].agent.contract,
        )
        self.assertTrue(
            all(
                request.agent.id == "qa-memory-worker"
                and "qa-memory-worker" in request.request_id
                for request in gateway.requests
            )
        )
        self.assertEqual(
            [
                HOTPOTQA_QA_MEMORY_TOOL_ID,
                HOTPOTQA_QA_MEMORY_TOOL_ID,
                HOTPOTQA_QA_MEMORY_TOOL_ID,
            ],
            [receipt["tool_id"] for receipt in response.metadata["tool_receipts"]],
        )
        self.assertTrue(
            all(
                entry["structured_action"].get("resource_id")
                in {HOTPOTQA_QA_MEMORY_TOOL_ID, None}
                for entry in response.metadata["react_trace"]
            )
        )
        serialized = json.dumps(dict(response.metadata), ensure_ascii=False).casefold()
        self.assertNotIn("web search", serialized)
        self.assertNotIn("web_search", serialized)

    def test_manifest_and_search_signature_mismatch_fails_closed(self) -> None:
        class _MismatchedIndex(_Index):
            manifest = _QAMemoryManifest()

        with self.assertRaisesRegex(
            TypeError, "QA-memory manifest and global search/read signatures differ"
        ):
            build_hotpotqa_embedding_tool_registry(
                _MismatchedIndex(),
                task_id="task-1",
                tool_id=HOTPOTQA_QA_MEMORY_TOOL_ID,
            )

    def test_factory_rejects_top_k_that_disagrees_with_manifest(self) -> None:
        with self.assertRaisesRegex(ValueError, "differs from the index manifest"):
            build_hotpotqa_embedding_tool_registry(
                _Index(), task_id="task-1", frozen_top_k=3
            )


if __name__ == "__main__":
    unittest.main()
