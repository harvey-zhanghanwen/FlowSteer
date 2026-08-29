from __future__ import annotations

import asyncio
from dataclasses import dataclass
import json
from typing import Any

from src.interactive.agent_graph import AgentGraph, AgentNode, AgentRelation
from src.interactive.agent_runtime import (
    AgentRequest,
    AgentResponse,
    AgentRuntime,
)
from src.interactive.agent_workflow_env import AgentWorkflowEnv
from src.interactive.director import AgentGraphOrchestrator
from src.interactive.hotpotqa_embedding_tool import (
    HOTPOTQA_FACT_MEMORY_TOOL_ID,
    HotpotQAEmbeddingReactExecutionAdapter,
    build_hotpotqa_embedding_tool_registry,
)
from src.interactive.hotpotqa_full_dataset_fact_memory_index import (
    FULL_DATASET_EVALUATION_SCOPE,
    FULL_DATASET_FACT_MEMORY_CORPUS_VERSION,
)
from src.interactive.model_registry import ModelRegistry, ModelSpec, ProviderSpec
from src.interactive.rollout_collector import SGLangReceiptDirectorClient
from src.interactive.tool_runtime import ToolRequest


@dataclass(frozen=True)
class _Manifest:
    schema_version: str = "flowsteer.hotpotqa.full_dataset_fact_memory_index.v1"
    index_id: str = "hotpotqa-full-dataset-fact-boundary-v1"
    corpus_version: str = FULL_DATASET_FACT_MEMORY_CORPUS_VERSION
    source: str = "HotpotQA declarative facts"
    source_splits: tuple[str, ...] = ("train", "validation")
    embedding_model: str = "test-encoder"
    embedding_dimension: int = 3
    normalized: bool = True
    similarity: str = "cosine"
    frozen_top_k: int = 1
    source_record_count: int = 97_852
    source_train_count: int = 90_447
    source_validation_count: int = 7_405
    unique_source_count: int = 97_852
    cycled_record_count: int = 0
    question_rewrite_count: int = 97_852
    fact_count: int = 97_852
    semantic_rewrite_coverage: float = 1.0
    frozen_evaluation_count: int = 128
    evaluation_overlap_count: int = 128
    contains_evaluation_source_facts: bool = True
    contains_raw_questions: bool = False
    contains_raw_answers: bool = False
    document_format: str = "declarative_fact_only"
    indexed_text_field: str = "fact_text"
    evaluation_scope: str = FULL_DATASET_EVALUATION_SCOPE
    official_heldout_eligible: bool = False
    paraphrase_versions: tuple[str, ...] = ("fact-v1",)
    paraphrase_provenances: tuple[str, ...] = ("unit-test",)


@dataclass(frozen=True)
class _RawHit:
    memory_id: str = "fact-1"
    fact_snippet: str = "Ada Lovelace authored the work."
    similarity: float = 0.97
    rank: int = 1
    # Simulate backend-private provenance accidentally attached to its object.
    # The Tool adapter must project none of it into the worker observation.
    original_question: str = "Who authored the work?"
    canonical_answer: str = "Ada Lovelace"
    paraphrase_question: str = "Which person wrote the work?"
    paraphrase_answer_statement: str = "The author is Ada Lovelace."


@dataclass(frozen=True)
class _RawFact:
    memory_id: str = "fact-1"
    fact_text: str = "Ada Lovelace authored the work."
    original_question: str = "Who authored the work?"
    canonical_answer: str = "Ada Lovelace"
    ground_truth: str = "Ada Lovelace"


class _Index:
    manifest = _Manifest()

    def __init__(self) -> None:
        self.search_calls: list[tuple[str, int]] = []
        self.read_calls: list[str] = []

    def search(self, query: str, k: int) -> tuple[_RawHit, ...]:
        self.search_calls.append((query, k))
        return (_RawHit(),)

    def read(self, memory_id: str) -> _RawFact:
        self.read_calls.append(memory_id)
        return _RawFact(memory_id=memory_id)


def _all_mapping_keys(value: Any) -> set[str]:
    if isinstance(value, dict):
        keys = {str(key) for key in value}
        for item in value.values():
            keys.update(_all_mapping_keys(item))
        return keys
    if isinstance(value, (list, tuple)):
        keys: set[str] = set()
        for item in value:
            keys.update(_all_mapping_keys(item))
        return keys
    return set()


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
        del prompt, seed
        raise AssertionError("this boundary test does not invoke the Director model")


class _DirectorTokenizer:
    def apply_chat_template(self, messages: object, **kwargs: object) -> list[int]:
        del messages, kwargs
        return [101, 102]

    def decode(self, token_ids: object, **kwargs: object) -> str:
        del token_ids, kwargs
        return ""


def test_fact_tool_projects_only_fact_fields_into_search_read_results_and_receipts() -> None:
    index = _Index()
    registry = build_hotpotqa_embedding_tool_registry(
        index,
        task_id="hotpotqa:evaluation-001",
        tool_id=HOTPOTQA_FACT_MEMORY_TOOL_ID,
    )

    search, search_receipt = asyncio.run(
        registry.ainvoke_with_receipt(
            HOTPOTQA_FACT_MEMORY_TOOL_ID,
            ToolRequest("search", {"query": "Who authored the work?", "k": 1}),
        )
    )
    assert search is not None
    assert set(search.value["hits"][0]) == {
        "memory_id",
        "fact_snippet",
        "similarity",
        "rank",
    }

    read, read_receipt = asyncio.run(
        registry.ainvoke_with_receipt(
            HOTPOTQA_FACT_MEMORY_TOOL_ID,
            ToolRequest("read", {"memory_id": "fact-1"}),
        )
    )
    assert read is not None
    assert set(read.value["fact"]) == {"memory_id", "fact_text"}
    forbidden = {
        "question",
        "original_question",
        "canonical_answer",
        "paraphrase_question",
        "paraphrase_answer_statement",
        "ground_truth",
        "answer",
        "evaluator_payload",
        "evaluator_receipt",
    }
    # The request-side `query` is intentionally retained as Tool provenance;
    # this assertion applies to corpus hits/read records and result receipts.
    for value in (
        search.value["hits"],
        read.value["fact"],
        search_receipt.to_value()["result"],
        read_receipt.to_value()["result"],
    ):
        assert forbidden.isdisjoint(_all_mapping_keys(value))
    assert "Question:" not in json.dumps(search.value["hits"])
    assert "Answer:" not in json.dumps(search.value["hits"])
    assert "Question:" not in json.dumps(read.value["fact"])
    assert "Answer:" not in json.dumps(read.value["fact"])
    assert index.search_calls == [("Who authored the work?", 1)]
    assert index.read_calls == ["fact-1"]


def test_director_has_no_tool_or_fact_payload_and_worker_receipts_route_by_relation() -> None:
    question_scope = "Who authored the work?"
    public_task = (
        "Based on the following passages, answer the question.\n\n"
        "[[Public] The work is a published book.\n\n"
        f"Question: {question_scope}"
    )
    index = _Index()
    tool_registry = build_hotpotqa_embedding_tool_registry(
        index,
        task_id="hotpotqa:evaluation-001",
        tool_id=HOTPOTQA_FACT_MEMORY_TOOL_ID,
    )
    gateway = _SequenceGateway(
        [
            _action(
                "tool",
                name="search",
                arguments={"query": question_scope, "k": 1},
                resource_id=HOTPOTQA_FACT_MEMORY_TOOL_ID,
            ),
            _action(
                "tool",
                name="read",
                arguments={"memory_id": "fact-1"},
                resource_id=HOTPOTQA_FACT_MEMORY_TOOL_ID,
            ),
            _action(
                "complete",
                name="complete",
                arguments={
                    "value": {
                        "retrieval_sufficiency": "supported",
                        "selected_memory_id": "fact-1",
                    }
                },
                resource_id=None,
            ),
            "<answer>Ada Lovelace</answer>",
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
        max_turns=3,
        max_tool_calls=2,
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
                "Retrieve one relevant declarative fact.",
                execution_mode="react",
                allowed_tools=(HOTPOTQA_FACT_MEMORY_TOOL_ID,),
            ),
            AgentNode("output", "m", "Answer from routed upstream evidence."),
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
        required_evidence_tool_id=HOTPOTQA_FACT_MEMORY_TOOL_ID,
        require_evidence_relation=True,
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
                    "contract": "Answer only from routed upstream evidence.",
                }
            )
        )
    )
    assert step.accepted and step.execution is not None
    execution = step.execution
    director_prompt = director.build_prompt(env, 1, ())
    director_payload = SGLangReceiptDirectorClient(
        _DirectorTokenizer(),
        policy_version="test-policy",
    ).request_payload(director_prompt, seed=1)

    assert not (
        {"tools", "tool_choice", "allowed_tools", "retrieval", "documents"}
        & set(director_payload)
    )
    assert "fact-1" not in director_prompt
    assert "Ada Lovelace authored the work." not in director_prompt
    assert "fact_text" not in env.snapshot().last_feedback
    assert "Ada Lovelace authored the work." not in env.snapshot().last_feedback

    worker_call = next(
        call for call in execution.calls if call.request.agent.id == "worker"
    )
    output_call = next(
        call for call in execution.calls if call.request.agent.id == "output"
    )
    worker_receipts = worker_call.response.metadata["tool_receipts"]
    assert [receipt["request"]["action"] for receipt in worker_receipts] == [
        "search",
        "read",
    ]
    assert all(
        receipt["tool_id"] == HOTPOTQA_FACT_MEMORY_TOOL_ID
        for receipt in worker_receipts
    )
    assert [request.agent.id for request in gateway.requests[:3]] == [
        "worker",
        "worker",
        "worker",
    ]
    assert gateway.requests[-1].agent.id == "output"
    assert output_call.request.agent.allowed_tools == ()
    assert len(output_call.request.upstream) == 1
    routed = output_call.request.upstream[0]
    assert routed.source_agent_id == "worker"
    assert routed.target_agent_id == "output"
    assert len(routed.tool_receipts) == 2
    assert env.finish_admissibility()["admissible"] is True

