from __future__ import annotations

from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import pickle
import re
import tempfile
import unittest

from src.interactive.healthbench_tool_adapter import (
    HEALTHBENCH_MEDRAG_SEARCH_TOOL_ID,
    HEALTHBENCH_PROFESSIONAL_DATASET_SCOPE,
    MEDRAG_BM25_TOP_K,
    open_healthbench_medrag_tool_registry,
)
from src.interactive.tool_runtime import ToolRequest


FIXTURE_SOURCE_REVISION = "1" * 40


def _write_bm25_fixture(root: Path) -> tuple[str, ...]:
    documents = (
        "Aspirin irreversibly inhibits platelet aggregation and reduces thrombosis.",
        "Insulin lowers blood glucose in patients with diabetes mellitus.",
        "Aspirin can cause gastrointestinal bleeding and peptic ulcer disease.",
        "Warfarin anticoagulation requires INR monitoring and increases bleeding.",
    )
    inverted: defaultdict[str, list[tuple[int, int]]] = defaultdict(list)
    document_frequency: Counter[str] = Counter()
    document_lengths: list[int] = []
    for document_id, contents in enumerate(documents):
        term_frequency = Counter(re.findall(r"\b\w+\b", contents.lower()))
        document_lengths.append(sum(term_frequency.values()))
        for term, frequency in term_frequency.items():
            inverted[term].append((document_id, frequency))
        document_frequency.update(term_frequency)
    index = {
        "avg_dl": sum(document_lengths) / len(documents),
        "doc_lens": document_lengths,
        "idf": {
            term: math.log(
                1.0
                + (len(documents) - frequency + 0.5) / (frequency + 0.5)
            )
            for term, frequency in document_frequency.items()
        },
        "inverted_index": dict(inverted),
    }
    with (root / "bm25_index.pkl").open("wb") as output:
        pickle.dump(index, output)
    with (root / "all_chunks.jsonl").open("w", encoding="utf-8") as output:
        for contents in documents:
            output.write(json.dumps({"contents": contents}) + "\n")
    (root / ".source_revision").write_text(
        FIXTURE_SOURCE_REVISION + "\n", encoding="utf-8"
    )
    return documents


class HealthBenchToolAdapterTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.documents = _write_bm25_fixture(self.root)

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def _open(self):
        return open_healthbench_medrag_tool_registry(
            corpus_root=self.root,
            source_identity="unit-fixture-medrag-textbooks",
            expected_source_revision=FIXTURE_SOURCE_REVISION,
            expected_rows=len(self.documents),
            timeout_seconds=1.0,
        )

    async def test_search_preserves_upstream_bm25_ranking_and_frozen_identity(
        self,
    ) -> None:
        with self._open() as opened:
            result, receipt = await opened.registry.ainvoke_with_receipt(
                HEALTHBENCH_MEDRAG_SEARCH_TOOL_ID,
                ToolRequest("search", {"query": "aspirin gastrointestinal bleeding"}),
            )

        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual("search", result.value["operation"])
        self.assertEqual("aspirin gastrointestinal bleeding", result.value["query"])
        self.assertEqual(MEDRAG_BM25_TOP_K, result.value["top_k"])
        self.assertEqual(
            {
                "source": "unit-fixture-medrag-textbooks",
                "source_revision": FIXTURE_SOURCE_REVISION,
                "corpus_rows": 4,
                "retrieval_backend": "bm25",
            },
            result.value["frozen_corpus"],
        )
        ranked = result.value["ranked_chunks"]
        self.assertEqual("2", ranked[0]["chunk_id"])
        self.assertEqual(1, ranked[0]["rank"])
        self.assertIn("gastrointestinal bleeding", ranked[0]["text"].lower())
        self.assertGreaterEqual(ranked[0]["score"], 1.0)
        self.assertEqual(FIXTURE_SOURCE_REVISION, receipt.tool_version)
        self.assertGreaterEqual(receipt.latency_ms, 0.0)
        # The receipt is canonical JSON and contains no opaque runtime object.
        json.dumps(receipt.to_value(), allow_nan=False)

    async def test_tool_schema_rejects_task_gold_and_evaluator_payloads(self) -> None:
        forbidden_fields = (
            "task_id",
            "ground_truth",
            "rubric",
            "physician_response",
            "evaluator_payload",
        )
        with self._open() as opened:
            for forbidden_field in forbidden_fields:
                result, receipt = await opened.registry.ainvoke_with_receipt(
                    HEALTHBENCH_MEDRAG_SEARCH_TOOL_ID,
                    ToolRequest(
                        "search",
                        {
                            "query": "aspirin",
                            forbidden_field: "evaluator-only",
                        },
                    ),
                )
                self.assertIsNone(result)
                self.assertEqual("ValueError", receipt.error_type)

    def test_capability_is_healthbench_only_with_fixed_top_three_schema(self) -> None:
        with self._open() as opened:
            capability = opened.registry.require_capability(
                HEALTHBENCH_MEDRAG_SEARCH_TOOL_ID
            )

        self.assertEqual(
            HEALTHBENCH_PROFESSIONAL_DATASET_SCOPE,
            capability.dataset_scope,
        )
        self.assertEqual(("search",), capability.action_names)
        self.assertEqual(
            capability.input_schema, capability.action_schemas["search"]
        )
        self.assertTrue(capability.supports_dataset("healthbench_professional"))
        self.assertFalse(capability.supports_dataset("hotpotqa"))
        self.assertEqual(["query"], capability.input_schema["required"])
        self.assertNotIn("top_k", capability.input_schema["properties"])
        self.assertEqual(
            MEDRAG_BM25_TOP_K,
            capability.output_schema["properties"]["top_k"]["const"],
        )
        self.assertEqual("none", capability.side_effect)

    async def test_close_is_idempotent_and_fails_closed_on_later_search(self) -> None:
        opened = self._open()
        registry = opened.registry
        opened.close()
        opened.close()

        self.assertTrue(opened.closed)
        result, receipt = await registry.ainvoke_with_receipt(
            HEALTHBENCH_MEDRAG_SEARCH_TOOL_ID,
            ToolRequest("search", {"query": "aspirin"}),
        )
        self.assertIsNone(result)
        self.assertEqual("RuntimeError", receipt.error_type)

    def test_formal_resource_contract_rejects_revision_or_row_mismatch(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "source revision differs"):
            open_healthbench_medrag_tool_registry(
                corpus_root=self.root,
                source_identity="unit-fixture-medrag-textbooks",
                expected_source_revision="2" * 40,
                expected_rows=len(self.documents),
            )
        with self.assertRaisesRegex(RuntimeError, "snippet count differs"):
            open_healthbench_medrag_tool_registry(
                corpus_root=self.root,
                source_identity="unit-fixture-medrag-textbooks",
                expected_source_revision=FIXTURE_SOURCE_REVISION,
                expected_rows=len(self.documents) + 1,
            )


if __name__ == "__main__":
    unittest.main()
