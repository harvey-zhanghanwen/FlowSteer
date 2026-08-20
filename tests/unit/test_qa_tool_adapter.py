from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from src.interactive.agent_graph import AgentNode
from src.interactive.agent_runtime import AgentRequest, AgentResponse, ExecutionPhase
from src.interactive.model_registry import ModelSpec, ProviderSpec
from src.interactive.qa_tool_adapter import (
    QARetrievalReactExecutionAdapter,
    QA_RETRIEVAL_TOOL_ID,
    build_qa_tool_registry,
    open_qa_tool_registry,
)
from src.interactive.tool_runtime import ToolRequest


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


if __name__ == "__main__":
    unittest.main()
