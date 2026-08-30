from __future__ import annotations

from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import pickle
import re
import tempfile
import unittest

from src.interactive.agent_graph import AgentNode
from src.interactive.agent_runtime import (
    AgentRequest,
    AgentResponse,
    ExecutionPhase,
)
from src.interactive.healthbench_tool_adapter import (
    HEALTHBENCH_MEDRAG_SEARCH_TOOL_ID,
    HEALTHBENCH_PROFESSIONAL_DATASET_SCOPE,
    HealthBenchMedRAGReactExecutionAdapter,
    MEDRAG_BM25_TOP_K,
    open_healthbench_medrag_tool_registry,
)
from src.interactive.model_registry import ModelSpec, ProviderSpec
from src.interactive.tool_runtime import StructuredAction, ToolRequest


FIXTURE_SOURCE_REVISION = "1" * 40


def _structured_action(
    kind: str,
    *,
    name: str,
    arguments: object,
    resource_id: str | None,
) -> str:
    return json.dumps(
        {
            "arguments": arguments,
            "kind": kind,
            "name": name,
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
        return AgentResponse(
            self.outputs.pop(0),
            {"provider_request_id": len(self.requests)},
        )


def _react_request() -> AgentRequest:
    return AgentRequest(
        request_id="healthbench-tool-unit:single",
        run_id="healthbench-tool-unit",
        graph_revision=1,
        problem="Synthetic public healthcare conversation.",
        agent=AgentNode(
            "tool_agent",
            "fixture-model",
            "Use the public conversation and any retrieved textbook evidence "
            "to produce one complete assistant response.",
            allowed_tools=(HEALTHBENCH_MEDRAG_SEARCH_TOOL_ID,),
            execution_mode="react",
            artifact_type="assistant_response",
            completion_condition="Return one complete assistant response.",
        ),
        model=ModelSpec("fixture-model", "fixture-provider"),
        provider=ProviderSpec("fixture-provider", kind="test"),
        phase=ExecutionPhase.SINGLE,
    )


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
        for document_id, contents in enumerate(documents):
            output.write(
                json.dumps(
                    {
                        "contents": contents,
                        "id": f"fixture-{document_id}",
                        "title": f"Fixture Textbook {document_id}",
                    }
                )
                + "\n"
            )
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
        self.assertEqual("fixture-2", ranked[0]["document_id"])
        self.assertEqual("Fixture Textbook 2", ranked[0]["title"])
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
        query_description = capability.input_schema["properties"]["query"][
            "description"
        ]
        self.assertIn("synonym", query_description)
        self.assertIn("expanded abbreviation", query_description)
        self.assertEqual(
            MEDRAG_BM25_TOP_K,
            capability.output_schema["properties"]["top_k"]["const"],
        )
        self.assertEqual("none", capability.side_effect)

    def test_react_domain_finishes_after_evidence_and_blocks_repeated_query(
        self,
    ) -> None:
        gateway = type("Gateway", (), {"generate": lambda *_: None})()
        with self._open() as opened:
            adapter = HealthBenchMedRAGReactExecutionAdapter(
                gateway=gateway,
                tool_registry=opened.registry,
                max_turns=6,
                max_tool_calls=3,
            )
            initial = adapter._state_conditioned_action_domain(None, [])
            successful = {
                "observation_status": "success",
                "executed_action": {
                    "kind": "tool",
                    "name": "search",
                    "resource_id": HEALTHBENCH_MEDRAG_SEARCH_TOOL_ID,
                    "skill_id": None,
                    "arguments": {"query": "acute kidney injury"},
                },
                "result": {"ranked_chunks": [{"text": "evidence"}]},
            }
            after_evidence = adapter._state_conditioned_action_domain(
                None,
                [successful],
            )
            repeated = StructuredAction.from_value(
                {
                    "kind": "tool",
                    "name": "search",
                    "resource_id": HEALTHBENCH_MEDRAG_SEARCH_TOOL_ID,
                    "skill_id": None,
                    "arguments": {"query": "  ACUTE   kidney injury  "},
                }
            )
            repeated_error = adapter._tool_action_error(
                request=None,
                action=repeated,
                observations=[successful],
            )

        self.assertEqual(
            (
                frozenset({(HEALTHBENCH_MEDRAG_SEARCH_TOOL_ID, "search")}),
                True,
            ),
            initial,
        )
        self.assertEqual((frozenset(), True), after_evidence)
        self.assertEqual("duplicate_tool_request", repeated_error)

    def test_react_domain_allows_query_pivot_only_until_tool_budget(self) -> None:
        gateway = type("Gateway", (), {"generate": lambda *_: None})()
        with self._open() as opened:
            adapter = HealthBenchMedRAGReactExecutionAdapter(
                gateway=gateway,
                tool_registry=opened.registry,
                max_turns=6,
                max_tool_calls=2,
            )
            empty_observations = [
                {
                    "observation_status": "success",
                    "executed_action": {
                        "kind": "tool",
                        "name": "search",
                        "resource_id": HEALTHBENCH_MEDRAG_SEARCH_TOOL_ID,
                        "skill_id": None,
                        "arguments": {"query": query},
                    },
                    "result": {"ranked_chunks": []},
                }
                for query in ("AKI", "acute kidney injury")
            ]
            after_first = adapter._state_conditioned_action_domain(
                None,
                empty_observations[:1],
            )
            at_budget = adapter._state_conditioned_action_domain(
                None,
                empty_observations,
            )

        self.assertEqual(
            (
                frozenset({(HEALTHBENCH_MEDRAG_SEARCH_TOOL_ID, "search")}),
                True,
            ),
            after_first,
        )
        self.assertEqual((frozenset(), True), at_budget)

    async def test_runtime_rejects_different_query_after_nonempty_evidence(
        self,
    ) -> None:
        gateway = _SequenceGateway(
            [
                _structured_action(
                    "tool",
                    name="search",
                    arguments={"query": "aspirin gastrointestinal bleeding"},
                    resource_id=HEALTHBENCH_MEDRAG_SEARCH_TOOL_ID,
                ),
                _structured_action(
                    "tool",
                    name="search",
                    arguments={"query": "warfarin anticoagulation"},
                    resource_id=HEALTHBENCH_MEDRAG_SEARCH_TOOL_ID,
                ),
                _structured_action(
                    "complete",
                    name="complete",
                    arguments={"value": "Evidence-supported response."},
                    resource_id=None,
                ),
            ]
        )
        with self._open() as opened:
            response = await HealthBenchMedRAGReactExecutionAdapter(
                gateway=gateway,
                tool_registry=opened.registry,
                max_turns=3,
                max_tool_calls=3,
            ).execute(_react_request())

        self.assertEqual("Evidence-supported response.", response.text)
        self.assertEqual(1, response.metadata["tool_calls"])
        self.assertEqual(1, len(response.metadata["tool_receipts"]))
        rejected = response.metadata["react_trace"][1]
        self.assertEqual("schema_invalid", rejected["observation_status"])
        self.assertEqual(
            "state_action_not_admitted",
            rejected["public_error_code"],
        )
        self.assertNotIn("observation", rejected)

    async def test_runtime_allows_different_query_after_empty_result(self) -> None:
        gateway = _SequenceGateway(
            [
                _structured_action(
                    "tool",
                    name="search",
                    arguments={"query": "no_such_medical_term"},
                    resource_id=HEALTHBENCH_MEDRAG_SEARCH_TOOL_ID,
                ),
                _structured_action(
                    "tool",
                    name="search",
                    arguments={"query": "aspirin gastrointestinal bleeding"},
                    resource_id=HEALTHBENCH_MEDRAG_SEARCH_TOOL_ID,
                ),
                _structured_action(
                    "complete",
                    name="complete",
                    arguments={"value": "Pivoted evidence response."},
                    resource_id=None,
                ),
            ]
        )
        with self._open() as opened:
            response = await HealthBenchMedRAGReactExecutionAdapter(
                gateway=gateway,
                tool_registry=opened.registry,
                max_turns=3,
                max_tool_calls=2,
            ).execute(_react_request())

        self.assertEqual("Pivoted evidence response.", response.text)
        self.assertEqual(2, response.metadata["tool_calls"])
        self.assertEqual(2, len(response.metadata["tool_receipts"]))
        first_result = response.metadata["react_trace"][0]["observation"][
            "result"
        ]
        self.assertEqual([], first_result["ranked_chunks"])
        second_result = response.metadata["react_trace"][1]["observation"][
            "result"
        ]
        self.assertTrue(second_result["ranked_chunks"])

    async def test_runtime_rejects_different_query_after_tool_budget(self) -> None:
        gateway = _SequenceGateway(
            [
                _structured_action(
                    "tool",
                    name="search",
                    arguments={"query": "no_such_medical_term"},
                    resource_id=HEALTHBENCH_MEDRAG_SEARCH_TOOL_ID,
                ),
                _structured_action(
                    "tool",
                    name="search",
                    arguments={"query": "aspirin gastrointestinal bleeding"},
                    resource_id=HEALTHBENCH_MEDRAG_SEARCH_TOOL_ID,
                ),
                _structured_action(
                    "complete",
                    name="complete",
                    arguments={"value": "Budget-bounded response."},
                    resource_id=None,
                ),
            ]
        )
        with self._open() as opened:
            response = await HealthBenchMedRAGReactExecutionAdapter(
                gateway=gateway,
                tool_registry=opened.registry,
                max_turns=3,
                max_tool_calls=1,
            ).execute(_react_request())

        self.assertEqual("Budget-bounded response.", response.text)
        self.assertEqual(1, response.metadata["tool_calls"])
        self.assertEqual(1, len(response.metadata["tool_receipts"]))
        self.assertEqual(
            "state_action_not_admitted",
            response.metadata["react_trace"][1]["public_error_code"],
        )

    async def test_runtime_rejects_normalized_repeated_query_without_receipt(
        self,
    ) -> None:
        gateway = _SequenceGateway(
            [
                _structured_action(
                    "tool",
                    name="search",
                    arguments={"query": "no_such_medical_term"},
                    resource_id=HEALTHBENCH_MEDRAG_SEARCH_TOOL_ID,
                ),
                _structured_action(
                    "tool",
                    name="search",
                    arguments={"query": "  NO_SUCH_MEDICAL_TERM  "},
                    resource_id=HEALTHBENCH_MEDRAG_SEARCH_TOOL_ID,
                ),
                _structured_action(
                    "tool",
                    name="search",
                    arguments={"query": "aspirin gastrointestinal bleeding"},
                    resource_id=HEALTHBENCH_MEDRAG_SEARCH_TOOL_ID,
                ),
                _structured_action(
                    "complete",
                    name="complete",
                    arguments={"value": "Non-repeated response."},
                    resource_id=None,
                ),
            ]
        )
        with self._open() as opened:
            response = await HealthBenchMedRAGReactExecutionAdapter(
                gateway=gateway,
                tool_registry=opened.registry,
                max_turns=4,
                max_tool_calls=2,
            ).execute(_react_request())

        self.assertEqual("Non-repeated response.", response.text)
        self.assertEqual(2, response.metadata["tool_calls"])
        self.assertEqual(2, len(response.metadata["tool_receipts"]))
        self.assertEqual(
            "duplicate_tool_request",
            response.metadata["react_trace"][1]["public_error_code"],
        )

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
