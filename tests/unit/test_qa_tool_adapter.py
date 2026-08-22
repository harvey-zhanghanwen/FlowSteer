from __future__ import annotations

from dataclasses import dataclass, replace
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from src.interactive.agent_graph import AgentNode
from src.interactive.agent_runtime import AgentRequest, AgentResponse, ExecutionPhase
from src.interactive.model_registry import ModelSpec, ProviderSpec
from src.interactive.react_execution import ReactExecutionError
from src.interactive.qa_tool_adapter import (
    QARetrievalReactExecutionAdapter,
    QA_RETRIEVAL_TOOL_ID,
    build_qa_tool_registry,
    open_qa_tool_registry,
    open_provided_context_qa_tool_registry,
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

    def test_provided_context_factory_reuses_skillflow_builder_and_removes_temp_index(self) -> None:
        index = FakeIndex()
        captured: dict[str, object] = {}

        class FakeDocumentPassage:
            def __init__(self, **kwargs: object) -> None:
                self.__dict__.update(kwargs)

        class FakeRetrievalIndexClass:
            @classmethod
            def open(cls, path: Path) -> FakeIndex:
                captured["opened_path"] = path
                return index

        def build(path: Path, passages: tuple[object, ...], **kwargs: object) -> None:
            captured["built_path"] = path
            captured["passages"] = passages
            captured["identity"] = kwargs
            path.write_text("fixture", encoding="utf-8")

        module = SimpleNamespace(
            DocumentPassage=FakeDocumentPassage,
            RetrievalIndex=FakeRetrievalIndexClass,
            build_retrieval_index=build,
        )
        with patch(
            "src.interactive.qa_tool_adapter._load_retrieval_module",
            return_value=module,
        ):
            opened = open_provided_context_qa_tool_registry(
                ["[Widsith] Widsith preserves a list of kings."],
                skillflow_source="/tmp/skillflow-source",
            )
            built_path = captured["built_path"]
            assert isinstance(built_path, Path)
            self.assertTrue(built_path.is_file())
            supplied = captured["passages"]
            assert isinstance(supplied, tuple)
            self.assertEqual("Widsith", supplied[0].title)
            self.assertEqual(
                "Widsith preserves a list of kings.", supplied[0].text
            )
            self.assertEqual(
                "benchmark-provided-context",
                captured["identity"]["corpus_name"],
            )
            opened.close()

        self.assertTrue(index.closed)
        self.assertFalse(built_path.parent.exists())

    def test_multi_hop_adapter_injects_upstream_skillflow_guidance(self) -> None:
        request = AgentRequest(
            request_id="qa:contract",
            run_id="qa",
            graph_revision=1,
            problem="What book contains Widsith?",
            agent=AgentNode(
                "retriever",
                "model",
                "find and chain the relevant passages",
                allowed_tools=(QA_RETRIEVAL_TOOL_ID,),
                execution_mode="react",
            ),
            model=ModelSpec("model", "provider"),
            provider=ProviderSpec("provider", kind="test"),
            phase=ExecutionPhase.SINGLE,
        )
        adapter = QARetrievalReactExecutionAdapter(
            gateway=SimpleNamespace(generate=lambda request: None),
            tool_registry=build_qa_tool_registry(FakeIndex()),
            max_turns=3,
            max_tool_calls=2,
            task_type="multi_hop_qa",
        )

        rendered = adapter._contract(request, [])

        self.assertIn("Chain evidence across passages", rendered)
        self.assertIn("specific entity names (not the full question)", rendered)
        self.assertNotIn("ReAct only as the execution schedule", rendered)
        self.assertNotIn("subject/entity, predicate/relation", rendered)
        self.assertNotIn("unexpectedly equal", rendered)
        self.assertNotIn('"value":"final artifact"', rendered)
        self.assertIn(
            "The completed artifact required by the Agent contract",
            rendered,
        )

        format_predecessor = AgentRequest(
            request_id=request.request_id,
            run_id=request.run_id,
            graph_revision=request.graph_revision,
            problem=request.problem,
            agent=request.agent,
            model=request.model,
            provider=request.provider,
            phase=request.phase,
            is_format_predecessor=True,
        )
        rendered_predecessor = adapter._contract(format_predecessor, [])
        self.assertIn("Candidate answer:", rendered_predecessor)
        self.assertIn("Evidence:", rendered_predecessor)

    def test_reasoner_and_verifier_receive_distinct_completion_guidance(self) -> None:
        adapter = QARetrievalReactExecutionAdapter(
            gateway=SimpleNamespace(generate=lambda request: None),
            tool_registry=build_qa_tool_registry(FakeIndex()),
            max_turns=3,
            max_tool_calls=2,
            task_type="multi_hop_qa",
        )

        def role_request(role_family: str) -> AgentRequest:
            return AgentRequest(
                request_id=f"qa:{role_family}",
                run_id="qa",
                graph_revision=1,
                problem="Which entity has the larger value?",
                agent=AgentNode(
                    role_family,
                    "model",
                    "preserve the original question scope",
                    role_family=role_family,
                    allowed_tools=(QA_RETRIEVAL_TOOL_ID,),
                    execution_mode="react",
                ),
                model=ModelSpec("model", "provider"),
                provider=ProviderSpec("provider", kind="test"),
                phase=ExecutionPhase.SINGLE,
                is_format_predecessor=role_family == "verifier",
                semantic_protocol="hotpotqa_verified_answer_slot_v1",
            )

        reasoner = adapter._contract(role_request("reasoner"), [])
        verifier = adapter._contract(role_request("verifier"), [])

        self.assertIn("Evidence propositions", reasoner)
        self.assertIn("Determine the one semantic candidate", reasoner)
        self.assertIn("Do not replace the Reasoner's candidate", verifier)
        self.assertIn("four explicit boolean check fields", verifier)
        self.assertIn("Set supported only when all checks pass", verifier)
        self.assertNotIn("Determine the one semantic candidate", verifier)

        default_reasoner = replace(
            role_request("reasoner"),
            semantic_protocol="none",
        )
        default_contract = adapter._contract(default_reasoner, [])
        self.assertNotIn("Evidence propositions", default_contract)
        self.assertNotIn("unexpectedly equal", default_contract)

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

    async def test_react_rejects_initial_completion_then_continues_after_dispatch(self) -> None:
        index = FakeIndex()

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
                    action("complete", {"value": "unsupported answer"}),
                    action("search", {"query": "Ada Lovelace", "limit": 1}),
                    action("read", {"passage_id": "p1"}),
                    action("complete", {"value": "Ada Lovelace"}),
                ]
                self.requests: list[AgentRequest] = []

            async def generate(self, request: AgentRequest) -> AgentResponse:
                self.requests.append(request)
                return AgentResponse(self.outputs.pop(0))

        gateway = SequenceGateway()
        request = AgentRequest(
            request_id="qa:completion-admission",
            run_id="qa",
            graph_revision=1,
            problem="Who wrote the first published algorithm?",
            agent=AgentNode(
                "retriever",
                "model",
                "retrieve evidence and answer",
                allowed_tools=(QA_RETRIEVAL_TOOL_ID,),
                execution_mode="react",
            ),
            model=ModelSpec("model", "provider"),
            provider=ProviderSpec("provider", kind="test"),
            phase=ExecutionPhase.SINGLE,
        )

        response = await QARetrievalReactExecutionAdapter(
            gateway=gateway,
            tool_registry=build_qa_tool_registry(index),
            max_turns=4,
            max_tool_calls=2,
            completion_policy="required_evidence",
        ).execute(request)

        self.assertEqual("Ada Lovelace", response.text)
        self.assertEqual(2, response.metadata["tool_calls"])
        self.assertEqual(
            "qa_completion_requires_successful_read_evidence",
            response.metadata["react_trace"][0]["public_error_code"],
        )
        self.assertIn(
            "qa_completion_requires_successful_read_evidence",
            gateway.requests[1].agent.contract,
        )
        self.assertIn(
            "next action must be qa-retrieval search",
            gateway.requests[1].agent.contract,
        )
        self.assertIn(
            '"arguments":{"limit":3,"query":"Who wrote the first published algorithm?"}',
            gateway.requests[1].agent.contract,
        )
        self.assertIn(
            "next action must be qa-retrieval read",
            gateway.requests[2].agent.contract,
        )
        self.assertIn('"p1"', gateway.requests[2].agent.contract)
        self.assertIn(
            '"arguments":{"passage_id":"p1"}',
            gateway.requests[2].agent.contract,
        )
        self.assertIn(
            "next action may complete",
            gateway.requests[3].agent.contract,
        )
        self.assertIn("Do not use kind=completion", gateway.requests[3].agent.contract)
        search_domain = gateway.requests[1].agent.contract.split(
            "Currently admissible Tool action schemas", 1
        )[1].split("\nCompletion", 1)[0]
        read_domain = gateway.requests[2].agent.contract.split(
            "Currently admissible Tool action schemas", 1
        )[1].split("\nCompletion", 1)[0]
        completion_domain = gateway.requests[3].agent.contract.split(
            "Currently admissible Tool action schemas", 1
        )[1].split("\nCurrently admissible completion", 1)[0]
        self.assertIn('"name":"search"', search_domain)
        self.assertNotIn('"name":"read"', search_domain)
        self.assertIn("Completion is not admissible", gateway.requests[1].agent.contract)
        self.assertIn('"name":"read"', read_domain)
        self.assertNotIn('"name":"search"', read_domain)
        self.assertEqual(" (name and resource_id are exact; arguments must satisfy the shown JSON schema): []", completion_domain)
        self.assertIn(
            "Currently admissible completion schema",
            gateway.requests[3].agent.contract,
        )
        self.assertEqual([("Ada Lovelace", 1)], index.search_calls)
        self.assertEqual(["p1"], index.read_calls)

    async def test_required_evidence_action_domain_rejects_search_after_search(
        self,
    ) -> None:
        index = FakeIndex()

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
                    action("search", {"query": "Ada Lovelace", "limit": 1}),
                    action("search", {"query": "Ada Lovelace", "limit": 1}),
                    action("read", {"passage_id": "p1"}),
                    action("complete", {"value": "Ada Lovelace"}),
                ]

            async def generate(self, request: AgentRequest) -> AgentResponse:
                del request
                return AgentResponse(self.outputs.pop(0))

        request = AgentRequest(
            request_id="qa:state-conditioned-domain",
            run_id="qa",
            graph_revision=1,
            problem="Who wrote the first published algorithm?",
            agent=AgentNode(
                "retriever",
                "model",
                "retrieve evidence and answer",
                allowed_tools=(QA_RETRIEVAL_TOOL_ID,),
                execution_mode="react",
            ),
            model=ModelSpec("model", "provider"),
            provider=ProviderSpec("provider", kind="test"),
            phase=ExecutionPhase.SINGLE,
        )

        response = await QARetrievalReactExecutionAdapter(
            gateway=SequenceGateway(),
            tool_registry=build_qa_tool_registry(index),
            max_turns=4,
            max_tool_calls=2,
            completion_policy="required_evidence",
        ).execute(request)

        self.assertEqual("Ada Lovelace", response.text)
        self.assertEqual(
            "qa_required_evidence_next_action_read",
            response.metadata["react_trace"][1]["public_error_code"],
        )
        self.assertEqual([("Ada Lovelace", 1)], index.search_calls)
        self.assertEqual(["p1"], index.read_calls)

    async def test_react_failed_retrieval_receipt_admits_explicit_completion(self) -> None:
        class FailingIndex(FakeIndex):
            def search(self, query: str, *, limit: int) -> tuple[FakeHit, ...]:
                self.search_calls.append((query, limit))
                raise TimeoutError("public retrieval timeout")

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
                    action("search", {"query": "Ada Lovelace", "limit": 1}),
                    action("complete", {"value": "insufficient evidence"}),
                ]

            async def generate(self, request: AgentRequest) -> AgentResponse:
                del request
                return AgentResponse(self.outputs.pop(0))

        response = await QARetrievalReactExecutionAdapter(
            gateway=SequenceGateway(),
            tool_registry=build_qa_tool_registry(FailingIndex()),
            max_turns=2,
            max_tool_calls=1,
        ).execute(
            AgentRequest(
                request_id="qa:failed-retrieval",
                run_id="qa",
                graph_revision=1,
                problem="Who wrote the first published algorithm?",
                agent=AgentNode(
                    "retriever",
                    "model",
                    "retrieve evidence and answer",
                    allowed_tools=(QA_RETRIEVAL_TOOL_ID,),
                    execution_mode="react",
                ),
                model=ModelSpec("model", "provider"),
                provider=ProviderSpec("provider", kind="test"),
                phase=ExecutionPhase.SINGLE,
            )
        )

        self.assertEqual("insufficient evidence", response.text)
        self.assertEqual(1, response.metadata["tool_calls"])
        self.assertEqual("TimeoutError", response.metadata["tool_receipts"][0]["error_type"])

    async def test_required_evidence_rejects_failed_retrieval_receipt(self) -> None:
        class FailingIndex(FakeIndex):
            def search(self, query: str, *, limit: int) -> tuple[FakeHit, ...]:
                self.search_calls.append((query, limit))
                raise TimeoutError("public retrieval timeout")

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
                    action("search", {"query": "Ada Lovelace", "limit": 1}),
                    action("complete", {"value": "unsupported answer"}),
                ]

            async def generate(self, request: AgentRequest) -> AgentResponse:
                del request
                return AgentResponse(self.outputs.pop(0))

        with self.assertRaises(ReactExecutionError) as caught:
            await QARetrievalReactExecutionAdapter(
                gateway=SequenceGateway(),
                tool_registry=build_qa_tool_registry(FailingIndex()),
                max_turns=2,
                max_tool_calls=1,
                completion_policy="required_evidence",
            ).execute(
                AgentRequest(
                    request_id="qa:required-evidence-failure",
                    run_id="qa",
                    graph_revision=1,
                    problem="Who wrote the first published algorithm?",
                    agent=AgentNode(
                        "retriever",
                        "model",
                        "retrieve evidence and answer",
                        allowed_tools=(QA_RETRIEVAL_TOOL_ID,),
                        execution_mode="react",
                    ),
                    model=ModelSpec("model", "provider"),
                    provider=ProviderSpec("provider", kind="test"),
                    phase=ExecutionPhase.SINGLE,
                )
            )

        error = caught.exception
        self.assertEqual("TimeoutError", error.tool_receipts[0]["error_type"])
        self.assertEqual(
            "qa_completion_requires_successful_read_evidence",
            error.react_trace[-1]["public_error_code"],
        )

    async def test_react_direct_completion_remains_valid_when_dispatch_is_impossible(self) -> None:
        def complete(value: str) -> str:
            return json.dumps(
                {
                    "arguments": {"value": value},
                    "kind": "complete",
                    "name": "complete",
                    "resource_id": None,
                    "skill_id": None,
                }
            )

        class OneShotGateway:
            async def generate(self, request: AgentRequest) -> AgentResponse:
                del request
                return AgentResponse(complete("Ada Lovelace"))

        for allowed_tools, max_tool_calls in (
            ((QA_RETRIEVAL_TOOL_ID,), 0),
            ((), 2),
        ):
            with self.subTest(
                allowed_tools=allowed_tools,
                max_tool_calls=max_tool_calls,
            ):
                response = await QARetrievalReactExecutionAdapter(
                    gateway=OneShotGateway(),
                    tool_registry=build_qa_tool_registry(FakeIndex()),
                    max_turns=1,
                    max_tool_calls=max_tool_calls,
                    completion_policy="optional",
                ).execute(
                    AgentRequest(
                        request_id="qa:no-dispatch",
                        run_id="qa",
                        graph_revision=1,
                        problem="Who wrote the first published algorithm?",
                        agent=AgentNode(
                            "retriever",
                            "model",
                            "answer",
                            allowed_tools=allowed_tools,
                            execution_mode="react",
                        ),
                        model=ModelSpec("model", "provider"),
                        provider=ProviderSpec("provider", kind="test"),
                        phase=ExecutionPhase.SINGLE,
                    )
                )

                self.assertEqual("Ada Lovelace", response.text)
                self.assertEqual(0, response.metadata["tool_calls"])
                self.assertEqual(
                    "completed",
                    response.metadata["react_trace"][0]["observation_status"],
                )


if __name__ == "__main__":
    unittest.main()
