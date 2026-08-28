from __future__ import annotations

from dataclasses import dataclass
import json
from types import SimpleNamespace

from src.interactive.agent_graph import AgentNode
from src.interactive.agent_runtime import AgentRequest, ExecutionPhase
from src.interactive.model_registry import ModelSpec, ProviderSpec
from src.interactive.qa_tool_adapter import (
    QARetrievalReactExecutionAdapter,
    RETRIEVAL_FIRST_PARAMETRIC_FALLBACK_POLICY,
    TRIVIAQA_QA_MEMORY_TOOL_ID,
    build_qa_tool_registry,
)
from src.interactive.react_execution import ToolReactExecutionAdapter
from src.interactive.tool_runtime import ToolRegistry


def _request(*, allowed_tools: tuple[str, ...] = ()) -> AgentRequest:
    return AgentRequest(
        request_id="schema-guidance:1",
        run_id="schema-guidance",
        graph_revision=1,
        problem="Which author wrote the novel?",
        agent=AgentNode(
            "worker",
            "model",
            "answer the question",
            allowed_tools=allowed_tools,
            execution_mode="react",
        ),
        model=ModelSpec("model", "provider"),
        provider=ProviderSpec("provider", kind="test"),
        phase=ExecutionPhase.SINGLE,
    )


def test_generic_completion_guidance_uses_actual_default_schema() -> None:
    adapter = ToolReactExecutionAdapter(
        gateway=SimpleNamespace(generate=lambda request: None),
        tool_registry=ToolRegistry(()),
        max_turns=1,
        max_tool_calls=0,
    )
    item = _request()
    schema = adapter._completion_arguments_schema(item)
    contract = adapter._contract(item, [])

    assert json.dumps(
        schema,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ) in contract
    assert 'arguments={"value": ...}' not in contract
    assert '"required":["value"]' in contract


@dataclass(frozen=True)
class _Manifest:
    corpus_name: str = "triviaqa-all-qa-memory"
    corpus_version: str = "transductive-v1"
    index_id: str = "qa-memory-index-v1"
    format: str = "flowsteer.triviaqa.qa-memory-embedding-index.v1"
    retrieval_backend: str = "normalized-dot-product"
    tool_id: str = TRIVIAQA_QA_MEMORY_TOOL_ID
    frozen_top_k: int = 1
    tool_budget: object = None


class _Index:
    manifest = _Manifest()

    def search(self, query: str, *, limit: int) -> tuple[object, ...]:
        del query, limit
        return ()

    def read(self, memory_id: str) -> object:
        raise KeyError(memory_id)

    def close(self) -> None:
        return None


def test_qamemory_completion_guidance_exposes_all_four_required_fields() -> None:
    adapter = QARetrievalReactExecutionAdapter(
        gateway=SimpleNamespace(generate=lambda request: None),
        tool_registry=build_qa_tool_registry(_Index()),
        max_turns=3,
        max_tool_calls=2,
        task_type="factual_qa",
        completion_policy=RETRIEVAL_FIRST_PARAMETRIC_FALLBACK_POLICY,
        retrieval_tool_id=TRIVIAQA_QA_MEMORY_TOOL_ID,
    )
    item = _request(allowed_tools=(TRIVIAQA_QA_MEMORY_TOOL_ID,))
    observations = [
        {
            "observation_status": "success",
            "executed_action": {
                "kind": "tool",
                "name": "search",
                "resource_id": TRIVIAQA_QA_MEMORY_TOOL_ID,
                "skill_id": None,
                "arguments": {"query": item.problem, "limit": 1},
            },
            "result": {
                "operation": "search",
                "query": item.problem,
                "top_k": 1,
                "memory_ids": ["memory-001"],
                "passage_ids": ["memory-001"],
                "hits": [{"memory_id": "memory-001", "rank": 1}],
            },
        },
        {
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
                    "text": "The answer is Ada.",
                },
                "passage": {
                    "memory_id": "memory-001",
                    "text": "The answer is Ada.",
                },
            },
        },
    ]
    schema = adapter._completion_arguments_schema(item)
    assert schema["required"] == [
        "value",
        "evidence_sufficiency",
        "answer_source",
        "supporting_memory_ids",
    ]

    contract = adapter._contract(item, observations)
    serialized_schema = json.dumps(
        schema,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    assert serialized_schema in contract
    for field in schema["required"]:
        assert f'"{field}"' in contract
    assert 'arguments={"value": ...}' not in contract
