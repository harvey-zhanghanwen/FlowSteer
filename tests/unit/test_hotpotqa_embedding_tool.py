from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any
import unittest

from src.interactive.hotpotqa_embedding_tool import (
    HOTPOTQA_RETRIEVAL_TOOL_ID,
    HotpotQAEmbeddingReactExecutionAdapter,
    build_hotpotqa_embedding_tool_registry,
)
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
        action_schemas = {
            schema["title"]: schema for schema in capability.input_schema["oneOf"]
        }
        self.assertEqual({"read", "search"}, set(action_schemas))
        self.assertEqual(2, action_schemas["search"]["properties"]["k"]["const"])
        self.assertNotIn("task_id", action_schemas["search"]["properties"])
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

    def test_factory_rejects_top_k_that_disagrees_with_manifest(self) -> None:
        with self.assertRaisesRegex(ValueError, "differs from the index manifest"):
            build_hotpotqa_embedding_tool_registry(
                _Index(), task_id="task-1", frozen_top_k=3
            )


if __name__ == "__main__":
    unittest.main()
