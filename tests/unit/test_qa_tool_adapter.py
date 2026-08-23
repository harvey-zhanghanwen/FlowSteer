from __future__ import annotations

from dataclasses import dataclass, replace
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from src.interactive.agent_graph import AgentNode
from src.interactive.agent_runtime import (
    AgentRequest,
    AgentResponse,
    ExecutionPhase,
    UpstreamMessage,
)
from src.interactive.model_registry import ModelSpec, ProviderSpec
from src.interactive.react_execution import ReactExecutionError
from src.interactive.qa_tool_adapter import (
    QARetrievalReactExecutionAdapter,
    QAStructuredReasoningExecutionAdapter,
    QA_RETRIEVAL_TOOL_ID,
    QA_VERIFIED_ANSWER_LINEAGE_PROTOCOL,
    build_qa_tool_registry,
    open_qa_tool_registry,
    open_provided_context_qa_tool_registry,
)
from src.interactive.tool_runtime import ActionKind, StructuredAction, ToolRequest


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
    async def test_reasoning_reasoner_reuses_skillflow_completion_schema(self) -> None:
        class CaptureGateway:
            def __init__(self) -> None:
                self.requests: list[AgentRequest] = []

            async def generate(self, request: AgentRequest) -> AgentResponse:
                self.requests.append(request)
                return AgentResponse("{}")

        gateway = CaptureGateway()
        schema_source = QARetrievalReactExecutionAdapter(
            gateway=gateway,
            tool_registry=build_qa_tool_registry(FakeIndex()),
            max_turns=3,
            max_tool_calls=2,
            task_type="multi_hop_qa",
        )
        adapter = QAStructuredReasoningExecutionAdapter(
            gateway=gateway,
            schema_source=schema_source,
        )
        request = AgentRequest(
            request_id="qa:structured-reasoning",
            run_id="qa",
            graph_revision=1,
            problem="Which magazine was started first, A or B?",
            agent=AgentNode(
                "reasoner",
                "model",
                "align routed evidence to the requested answer slot",
                role_family="reasoner",
                execution_mode="reasoning",
            ),
            model=ModelSpec("model", "provider"),
            provider=ProviderSpec("provider", kind="test"),
            phase=ExecutionPhase.SINGLE,
            semantic_protocol="hotpotqa_role_conditional_capabilities_v1",
        )

        await adapter.execute(request)

        schema = json.loads(
            gateway.requests[0].model.metadata["response_json_schema"]
        )
        self.assertEqual(
            {
                "question_scope",
                "answer_slot",
                "evidence_propositions",
                "multi_hop_chain",
                "candidate_answer",
                "evidence",
            },
            set(schema["required"]),
        )
        self.assertEqual(
            "entity",
            schema["properties"]["answer_slot"]["properties"]
            ["answer_type"]["const"],
        )
        self.assertEqual(
            "single",
            schema["properties"]["answer_slot"]["properties"]
            ["answer_cardinality"]["const"],
        )
        self.assertEqual(
            2,
            schema["properties"]["evidence_propositions"]["minItems"],
        )
        self.assertEqual(
            2,
            schema["properties"]["multi_hop_chain"]["minItems"],
        )

        await adapter.execute(
            replace(
                request,
                agent=replace(
                    request.agent,
                    id="output",
                    role_family="output",
                ),
            )
        )
        self.assertNotIn(
            "response_json_schema",
            gateway.requests[1].model.metadata,
        )

    async def test_model_visible_continuation_collapses_only_duplicate_errors(
        self,
    ) -> None:
        error = {
            "observation_status": "schema_invalid",
            "public_error_code": (
                "qa_semantic_artifact_invalid: repair answer binding"
            ),
        }
        success = {
            "observation_status": "success",
            "result": {"operation": "read", "passage_id": "p1"},
        }

        visible = QARetrievalReactExecutionAdapter._model_visible_observations(
            [error, dict(error), success, error, dict(error)]
        )

        self.assertEqual(3, len(visible))
        self.assertEqual("success", visible[1]["observation_status"])
        self.assertIn("repair_instruction", visible[0])
        self.assertIn("repair_instruction", visible[2])
        self.assertEqual(2, visible[0]["repeat_count"])
        self.assertEqual(2, visible[2]["repeat_count"])

    async def test_duplicate_normalized_query_exposes_query_rewrite_feedback(
        self,
    ) -> None:
        visible = QARetrievalReactExecutionAdapter._model_visible_observations(
            [
                {
                    "observation_status": "schema_invalid",
                    "public_error_code": (
                        "qa_retrieval_duplicate_normalized_query"
                    ),
                }
            ]
        )

        self.assertEqual(1, len(visible))
        self.assertIn(
            "semantically distinct entity-and-relation query",
            visible[0]["repair_instruction"],
        )
        self.assertIn("do not repeat", visible[0]["repair_instruction"])

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

    def test_free_agent_exact_answer_schema_is_terminal_output_only(self) -> None:
        class CompletionOnlyAdapter(QARetrievalReactExecutionAdapter):
            def _state_conditioned_action_domain(
                self,
                request: AgentRequest,
                observations: list[dict[str, object]],
            ) -> tuple[frozenset[tuple[str, str]], bool]:
                del request, observations
                return frozenset(), True

        adapter = CompletionOnlyAdapter(
            gateway=SimpleNamespace(generate=lambda request: None),
            tool_registry=build_qa_tool_registry(FakeIndex()),
            max_turns=3,
            max_tool_calls=2,
            task_type="multi_hop_qa",
        )
        output_request = AgentRequest(
            request_id="qa:free-output",
            run_id="qa",
            graph_revision=1,
            problem="What book contains Widsith?",
            agent=AgentNode(
                "synthesis",
                "model",
                "answer from the available evidence",
                role_family="synthesis",
                allowed_tools=(QA_RETRIEVAL_TOOL_ID,),
                execution_mode="react",
            ),
            model=ModelSpec("model", "provider"),
            provider=ProviderSpec("provider", kind="test"),
            phase=ExecutionPhase.SINGLE,
            is_output_agent=True,
            require_exact_answer_tag=True,
        )

        response_schema = adapter._state_conditioned_response_schema(
            output_request,
            [],
        )
        self.assertIsNotNone(response_schema)
        assert response_schema is not None
        value_schema = response_schema["properties"]["arguments"]["properties"][
            "value"
        ]
        self.assertEqual("string", value_schema["type"])
        self.assertIn(
            "Exactly one non-empty <answer>...</answer> wrapper",
            value_schema["description"],
        )
        self.assertEqual(
            {"value"},
            set(response_schema["properties"]["arguments"]["properties"]),
        )
        self.assertNotIn(
            "<answer>",
            json.dumps(
                {
                    key: value
                    for key, value in response_schema["properties"].items()
                    if key != "arguments"
                },
                sort_keys=True,
            ),
        )

        output_contract = adapter._contract(output_request, [])
        self.assertIn('arguments={"value": ...}', output_contract)
        self.assertIn(
            "Exactly one non-empty <answer>...</answer> wrapper",
            output_contract,
        )

        intermediate_request = replace(
            output_request,
            request_id="qa:free-intermediate",
            is_output_agent=False,
        )
        intermediate_schema = adapter._completion_arguments_schema(
            intermediate_request
        )
        self.assertNotIn("<answer>", json.dumps(intermediate_schema, sort_keys=True))
        self.assertNotIn(
            "<answer>",
            adapter._contract(intermediate_request, []),
        )

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

        self.assertIn("evidence_propositions", reasoner)
        self.assertIn("bind the candidate", reasoner)
        self.assertIn(
            "minimal but complete evidence-aligned referential surface",
            reasoner,
        )
        self.assertIn("possessive marker", reasoner)
        self.assertIn("complete possessor entity mention", reasoner)
        self.assertIn("antecedent-bearing span", reasoner)
        self.assertIn("unknown, not zero", reasoner)
        self.assertIn("narrower subtype", reasoner)
        self.assertIn("Do not replace the Reasoner's candidate", verifier)
        self.assertIn("seven explicit boolean check fields", verifier)
        self.assertIn("Set supported only when all checks pass", verifier)
        self.assertIn(
            "minimal but complete evidence-aligned referential surface",
            verifier,
        )
        self.assertIn("drops part of the possessor entity mention", verifier)
        self.assertNotIn("bind the candidate", verifier)

        completion_schema = adapter._completion_arguments_schema(
            role_request("reasoner")
        )
        semantic_properties = completion_schema["properties"]["value"]["properties"]
        proposition_properties = semantic_properties["evidence_propositions"][
            "items"
        ]["properties"]
        for field in ("subject", "object_or_attribute_value"):
            description = proposition_properties[field]["description"]
            self.assertIn(
                "minimal but complete evidence-aligned referential surface",
                description,
            )
            self.assertIn("complete possessor mention", description)
            self.assertIn("possessive marker", description)
            self.assertIn("name suffix", description)
        candidate_description = semantic_properties["candidate_answer"][
            "description"
        ]
        self.assertIn(
            "minimal but complete evidence-aligned referential surface",
            candidate_description,
        )
        self.assertIn("not a strict subspan", candidate_description)

        default_reasoner = replace(
            role_request("reasoner"),
            semantic_protocol="none",
        )
        default_contract = adapter._contract(default_reasoner, [])
        self.assertNotIn("evidence_propositions", default_contract)
        self.assertNotIn("unexpectedly equal", default_contract)

    def test_role_conditional_react_output_copies_routed_candidate(self) -> None:
        class CompletionOnlyAdapter(QARetrievalReactExecutionAdapter):
            def _state_conditioned_action_domain(
                self,
                request: AgentRequest,
                observations: list[dict[str, object]],
            ) -> tuple[frozenset[tuple[str, str]], bool]:
                del request, observations
                return frozenset(), True

        adapter = CompletionOnlyAdapter(
            gateway=SimpleNamespace(generate=lambda request: None),
            tool_registry=build_qa_tool_registry(FakeIndex()),
            max_turns=3,
            max_tool_calls=2,
            task_type="multi_hop_qa",
        )
        request = AgentRequest(
            request_id="hotpot:react-output-candidate",
            run_id="hotpot",
            graph_revision=1,
            problem="Which target is reached through Bridge Beta?",
            agent=AgentNode(
                "output",
                "model",
                "return the terminal task answer",
                role_family="output",
                allowed_tools=(QA_RETRIEVAL_TOOL_ID,),
                execution_mode="react",
            ),
            model=ModelSpec("model", "provider"),
            provider=ProviderSpec("provider", kind="test"),
            phase=ExecutionPhase.SINGLE,
            is_output_agent=True,
            require_exact_answer_tag=True,
            semantic_protocol="hotpotqa_role_conditional_capabilities_v1",
            upstream=(
                UpstreamMessage(
                    "semantic",
                    "output",
                    json.dumps({"Candidate answer": "Target Gamma"}),
                    artifact_type="semantic_candidate",
                ),
            ),
        )

        schema = adapter._completion_arguments_schema(request)
        self.assertEqual(
            "<answer>Target Gamma</answer>",
            schema["properties"]["value"]["const"],
        )
        contract = adapter._contract(request, [])
        self.assertIn(
            "Set arguments.value to exactly <answer>Target Gamma</answer>",
            contract,
        )
        self.assertIn("do not reselect, canonicalize, or rewrite", contract)

    def test_required_evidence_shows_reasoner_artifact_only_at_completion_state(
        self,
    ) -> None:
        adapter = QARetrievalReactExecutionAdapter(
            gateway=SimpleNamespace(generate=lambda request: None),
            tool_registry=build_qa_tool_registry(FakeIndex()),
            max_turns=4,
            max_tool_calls=2,
            task_type="multi_hop_qa",
            completion_policy="required_evidence",
        )
        request = AgentRequest(
            request_id="qa:semantic-stage",
            run_id="qa",
            graph_revision=1,
            problem="Which city is requested?",
            agent=AgentNode(
                "reasoner",
                "model",
                "preserve scope and determine the semantic answer",
                role_family="reasoner",
                allowed_tools=(QA_RETRIEVAL_TOOL_ID,),
                execution_mode="react",
            ),
            model=ModelSpec("model", "provider"),
            provider=ProviderSpec("provider", kind="test"),
            phase=ExecutionPhase.SINGLE,
            semantic_protocol="hotpotqa_verified_answer_slot_v1",
        )
        search_observation = {
            "observation_status": "success",
            "result": {
                "operation": "search",
                "passage_ids": ["p1"],
            },
        }
        read_observation = {
            "observation_status": "success",
            "result": {
                "operation": "read",
                "passage_id": "p1",
                "passage": {"text": "Paris is the capital of France."},
            },
        }

        search_contract = adapter._contract(request, [])
        read_contract = adapter._contract(request, [search_observation])
        completion_contract = adapter._contract(
            request,
            [search_observation, read_observation],
        )

        self.assertNotIn("six structured fields", search_contract)
        self.assertNotIn("six structured fields", read_contract)
        self.assertIn("six structured fields", completion_contract)
        self.assertNotIn('"arguments":{"type":"object"', search_contract)
        self.assertNotIn('"arguments":{"type":"object"', read_contract)
        self.assertIn("Completion is not admitted in this Tool-only state", search_contract)
        self.assertIn("Completion is not admitted in this Tool-only state", read_contract)

    def test_unified_factual_schema_uses_one_grounded_proposition(self) -> None:
        adapter = QARetrievalReactExecutionAdapter(
            gateway=SimpleNamespace(generate=lambda request: None),
            tool_registry=build_qa_tool_registry(FakeIndex()),
            max_turns=6,
            max_tool_calls=10,
            task_type="factual_qa",
            completion_policy="required_evidence",
        )
        request = AgentRequest(
            request_id="trivia:semantic-schema",
            run_id="trivia",
            graph_revision=1,
            problem="Where was the historical figure born?",
            agent=AgentNode(
                "reasoner",
                "model",
                "retrieve evidence and bind the requested relation",
                role_family="reasoner",
                allowed_tools=(QA_RETRIEVAL_TOOL_ID,),
                execution_mode="react",
            ),
            model=ModelSpec("model", "provider"),
            provider=ProviderSpec("provider", kind="test"),
            phase=ExecutionPhase.SINGLE,
            semantic_protocol=QA_VERIFIED_ANSWER_LINEAGE_PROTOCOL,
        )

        schema = adapter._completion_arguments_schema(request)
        fields = schema["properties"]["value"]["properties"]
        self.assertEqual(
            "Where was the historical figure born?",
            fields["question_scope"]["const"],
        )
        self.assertEqual(
            "location",
            fields["answer_slot"]["properties"]["answer_type"]["const"],
        )
        self.assertEqual(1, fields["evidence_propositions"]["minItems"])
        self.assertEqual(1, fields["multi_hop_chain"]["minItems"])
        self.assertIn(
            "explicit identity binding",
            fields["evidence_propositions"]["items"]["properties"]
            ["subject"]["description"],
        )

        initial_schema = adapter._state_conditioned_response_schema(request, [])
        assert initial_schema is not None
        self.assertEqual(
            5,
            initial_schema["properties"]["arguments"]["properties"]
            ["limit"]["const"],
        )
        initial_contract = adapter._contract(request, [])
        self.assertIn("`initial_retrieval`", initial_contract)
        self.assertIn("entity identity and target relation", initial_contract)

    async def test_unified_factual_completion_accepts_one_grounded_proposition(
        self,
    ) -> None:
        class FactualIndex(FakeIndex):
            def read(self, passage_id: str) -> FakePassage:
                self.read_calls.append(passage_id)
                return FakePassage(
                    passage_id,
                    "d1",
                    "France",
                    "Paris is the capital of France.",
                )

        artifact = {
            "question_scope": "What is the capital of France?",
            "answer_slot": {
                "answer_type": "short_answer",
                "answer_cardinality": "single",
                "qualifiers": [],
                "proposition_index": 0,
                "answer_field": "object_or_attribute_value",
            },
            "evidence_propositions": [
                {
                    "subject": "France",
                    "relation": "capital",
                    "object_or_attribute_value": "Paris",
                    "qualifiers": [],
                    "evidence_span": "Paris is the capital of France.",
                }
            ],
            "multi_hop_chain": ["France --capital--> Paris"],
            "candidate_answer": "Paris",
            "evidence": ["Paris is the capital of France."],
        }

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
                    action("search", {"query": "France capital", "limit": 5}),
                    action("read", {"passage_id": "p1"}),
                    action("complete", {"value": artifact}),
                ]

            async def generate(self, request: AgentRequest) -> AgentResponse:
                del request
                return AgentResponse(self.outputs.pop(0))

        index = FactualIndex()
        request = AgentRequest(
            request_id="trivia:one-proposition",
            run_id="trivia",
            graph_revision=1,
            problem="What is the capital of France?",
            agent=AgentNode(
                "reasoner",
                "model",
                "retrieve and bind the factual answer",
                role_family="reasoner",
                allowed_tools=(QA_RETRIEVAL_TOOL_ID,),
                execution_mode="react",
            ),
            model=ModelSpec("model", "provider"),
            provider=ProviderSpec("provider", kind="test"),
            phase=ExecutionPhase.SINGLE,
            semantic_protocol=QA_VERIFIED_ANSWER_LINEAGE_PROTOCOL,
        )

        response = await QARetrievalReactExecutionAdapter(
            gateway=SequenceGateway(),
            tool_registry=build_qa_tool_registry(index),
            max_turns=3,
            max_tool_calls=2,
            task_type="factual_qa",
            completion_policy="required_evidence",
        ).execute(request)

        self.assertEqual(
            json.dumps(
                artifact,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ),
            response.text,
        )
        self.assertEqual([("France capital", 5)], index.search_calls)
        self.assertEqual(["p1"], index.read_calls)

    def test_unified_factual_evidence_rejection_advances_retrieval_strategy(
        self,
    ) -> None:
        adapter = QARetrievalReactExecutionAdapter(
            gateway=SimpleNamespace(generate=lambda request: None),
            tool_registry=build_qa_tool_registry(FakeIndex()),
            max_turns=8,
            max_tool_calls=10,
            task_type="factual_qa",
            completion_policy="required_evidence",
        )
        request = AgentRequest(
            request_id="trivia:retry",
            run_id="trivia",
            graph_revision=1,
            problem="Who wrote the novel?",
            agent=AgentNode(
                "reasoner",
                "model",
                "retrieve evidence and bind the requested relation",
                role_family="reasoner",
                allowed_tools=(QA_RETRIEVAL_TOOL_ID,),
                execution_mode="react",
            ),
            model=ModelSpec("model", "provider"),
            provider=ProviderSpec("provider", kind="test"),
            phase=ExecutionPhase.SINGLE,
            semantic_protocol=QA_VERIFIED_ANSWER_LINEAGE_PROTOCOL,
        )
        search = {
            "observation_status": "success",
            "executed_action": {
                "kind": "tool",
                "name": "search",
                "resource_id": QA_RETRIEVAL_TOOL_ID,
                "skill_id": None,
                "arguments": {"query": "Novel author", "limit": 5},
            },
            "result": {
                "operation": "search",
                "query": "Novel author",
                "top_k": 5,
                "passage_ids": ["p1"],
            },
        }
        read = {
            "observation_status": "success",
            "executed_action": {
                "kind": "tool",
                "name": "read",
                "resource_id": QA_RETRIEVAL_TOOL_ID,
                "skill_id": None,
                "arguments": {"passage_id": "p1"},
            },
            "result": {
                "operation": "read",
                "passage_id": "p1",
                "passage": {"text": "A different entity is discussed here."},
            },
        }
        rejection = {
            "observation_status": "schema_invalid",
            "public_error_code": (
                "qa_semantic_evidence_provenance_invalid: target relation absent"
            ),
        }
        observations = [search, read, rejection]

        admitted, completion = adapter._state_conditioned_action_domain(
            request,
            observations,
        )
        self.assertEqual(
            frozenset({(QA_RETRIEVAL_TOOL_ID, "search")}), admitted
        )
        self.assertFalse(completion)
        retry_schema = adapter._state_conditioned_response_schema(
            request,
            observations,
        )
        assert retry_schema is not None
        self.assertEqual(
            10,
            retry_schema["properties"]["arguments"]["properties"]
            ["limit"]["const"],
        )
        retry_contract = adapter._contract(request, observations)
        self.assertIn("`spelling_normalization`", retry_contract)
        self.assertIn('"novel author"', retry_contract)

        repeated = StructuredAction(
            ActionKind.TOOL,
            "search",
            {"query": "  NOVEL   AUTHOR ", "limit": 10},
            resource_id=QA_RETRIEVAL_TOOL_ID,
        )
        self.assertEqual(
            "qa_retrieval_duplicate_normalized_query",
            adapter._tool_action_error(
                request=request,
                action=repeated,
                observations=observations,
            ),
        )

        retry_search = {
            "observation_status": "success",
            "executed_action": {
                "kind": "tool",
                "name": "search",
                "resource_id": QA_RETRIEVAL_TOOL_ID,
                "skill_id": None,
                "arguments": {"query": "Novelist pen name", "limit": 10},
            },
            "result": {
                "operation": "search",
                "query": "Novelist pen name",
                "top_k": 10,
                "passage_ids": ["p2"],
            },
        }
        admitted_after_retry, completion_after_retry = (
            adapter._state_conditioned_action_domain(
                request,
                [*observations, retry_search],
            )
        )
        self.assertEqual(
            frozenset({(QA_RETRIEVAL_TOOL_ID, "read")}),
            admitted_after_retry,
        )
        self.assertFalse(completion_after_retry)

    def test_unified_factual_upstream_read_does_not_mask_evidence_recovery(
        self,
    ) -> None:
        adapter = QARetrievalReactExecutionAdapter(
            gateway=SimpleNamespace(generate=lambda request: None),
            tool_registry=build_qa_tool_registry(FakeIndex()),
            max_turns=8,
            max_tool_calls=10,
            task_type="factual_qa",
            completion_policy="required_evidence",
        )
        upstream_read_receipt = {
            "tool_id": QA_RETRIEVAL_TOOL_ID,
            "request": {
                "action": "read",
                "arguments": {"passage_id": "upstream-p1"},
            },
            "result": {
                "value": {
                    "operation": "read",
                    "passage": {
                        "passage_id": "upstream-p1",
                        "text": "This passage discusses a different entity.",
                    },
                }
            },
            "error_type": None,
        }
        request = AgentRequest(
            request_id="trivia:upstream-recovery",
            run_id="trivia",
            graph_revision=2,
            problem="Where was the actor born?",
            agent=AgentNode(
                "reasoner",
                "model",
                "bind retrieved evidence to the requested relation",
                role_family="reasoner",
                allowed_tools=(QA_RETRIEVAL_TOOL_ID,),
                execution_mode="react",
            ),
            model=ModelSpec("model", "provider"),
            provider=ProviderSpec("provider", kind="test"),
            phase=ExecutionPhase.SINGLE,
            semantic_protocol=QA_VERIFIED_ANSWER_LINEAGE_PROTOCOL,
            upstream=(
                UpstreamMessage(
                    "retriever",
                    "reasoner",
                    "retrieved upstream-p1",
                    graph_revision=2,
                    artifact_type="retrieval_evidence",
                    tool_receipts=(upstream_read_receipt,),
                ),
            ),
        )

        initial_actions, initial_completion = (
            adapter._state_conditioned_action_domain(request, [])
        )
        self.assertEqual(
            frozenset({(QA_RETRIEVAL_TOOL_ID, "search")}),
            initial_actions,
        )
        self.assertTrue(initial_completion)

        rejection = {
            "observation_status": "schema_invalid",
            "public_error_code": (
                "qa_semantic_evidence_provenance_invalid: target relation absent"
            ),
        }
        recovery_actions, recovery_completion = (
            adapter._state_conditioned_action_domain(request, [rejection])
        )
        self.assertEqual(
            frozenset({(QA_RETRIEVAL_TOOL_ID, "search")}),
            recovery_actions,
        )
        self.assertFalse(recovery_completion)

        own_search = {
            "observation_status": "success",
            "executed_action": {
                "kind": "tool",
                "name": "search",
                "resource_id": QA_RETRIEVAL_TOOL_ID,
                "skill_id": None,
                "arguments": {"query": "actor birthplace", "limit": 5},
            },
            "result": {
                "operation": "search",
                "query": "actor birthplace",
                "top_k": 5,
                "passage_ids": ["own-p1"],
            },
        }
        own_read = {
            "observation_status": "success",
            "executed_action": {
                "kind": "tool",
                "name": "read",
                "resource_id": QA_RETRIEVAL_TOOL_ID,
                "skill_id": None,
                "arguments": {"passage_id": "own-p1"},
            },
            "result": {
                "operation": "read",
                "passage_id": "own-p1",
                "passage": {"text": "The actor was born in the city."},
            },
        }
        final_actions, final_completion = (
            adapter._state_conditioned_action_domain(
                request,
                [rejection, own_search, own_read],
            )
        )
        self.assertEqual(frozenset(), final_actions)
        self.assertTrue(final_completion)

    async def test_unified_factual_exhaustion_reports_coverage_failure(
        self,
    ) -> None:
        class EmptyIndex(FakeIndex):
            def search(self, query: str, *, limit: int) -> tuple[FakeHit, ...]:
                self.search_calls.append((query, limit))
                return ()

        def search_action(query: str, limit: int) -> str:
            return json.dumps(
                {
                    "arguments": {"query": query, "limit": limit},
                    "kind": "tool",
                    "name": "search",
                    "resource_id": QA_RETRIEVAL_TOOL_ID,
                    "skill_id": None,
                }
            )

        class SequenceGateway:
            def __init__(self) -> None:
                self.outputs = [
                    search_action("Entity relation", 5),
                    search_action("Entitiy relation", 10),
                    search_action("Entity alias relation", 15),
                    search_action("Entity qualifier relation", 20),
                    search_action("Relation of entity", 25),
                    json.dumps(
                        {
                            "arguments": {"value": "unsupported guess"},
                            "kind": "complete",
                            "name": "complete",
                            "resource_id": None,
                            "skill_id": None,
                        }
                    ),
                ]

            async def generate(self, request: AgentRequest) -> AgentResponse:
                del request
                return AgentResponse(self.outputs.pop(0))

        request = AgentRequest(
            request_id="trivia:coverage",
            run_id="trivia",
            graph_revision=1,
            problem="Who held the office?",
            agent=AgentNode(
                "reasoner",
                "model",
                "retrieve evidence before answering",
                role_family="reasoner",
                allowed_tools=(QA_RETRIEVAL_TOOL_ID,),
                execution_mode="react",
            ),
            model=ModelSpec("model", "provider"),
            provider=ProviderSpec("provider", kind="test"),
            phase=ExecutionPhase.SINGLE,
            semantic_protocol=QA_VERIFIED_ANSWER_LINEAGE_PROTOCOL,
        )

        with self.assertRaises(ReactExecutionError) as caught:
            await QARetrievalReactExecutionAdapter(
                gateway=SequenceGateway(),
                tool_registry=build_qa_tool_registry(EmptyIndex()),
                max_turns=6,
                max_tool_calls=10,
                task_type="factual_qa",
                completion_policy="required_evidence",
            ).execute(request)

        error = caught.exception
        self.assertEqual(5, len(error.tool_receipts))
        self.assertIs(True, error.tool_plan_exhausted)
        self.assertEqual(
            "knowledge_base_coverage_failure",
            error.react_trace[-1]["public_error_code"],
        )
        terminal_diagnosis = error.react_trace[-1][
            "terminal_failure_diagnosis"
        ]
        self.assertIs(True, terminal_diagnosis["tool_plan_exhausted"])
        self.assertEqual(5, terminal_diagnosis["retrieval_attempt_count"])
        self.assertEqual(
            [5, 10, 15, 20, 25],
            terminal_diagnosis["search_top_ks"],
        )

    async def test_unified_factual_exhaustion_after_reads_keeps_coverage_receipt(
        self,
    ) -> None:
        class MismatchedIndex(FakeIndex):
            def search(self, query: str, *, limit: int) -> tuple[FakeHit, ...]:
                self.search_calls.append((query, limit))
                passage_id = f"p{len(self.search_calls)}"
                return (
                    FakeHit(
                        passage_id,
                        f"d{len(self.search_calls)}",
                        "Different entity",
                        "Evidence for another relation.",
                        1,
                    ),
                )

            def read(self, passage_id: str) -> FakePassage:
                self.read_calls.append(passage_id)
                return FakePassage(
                    passage_id,
                    passage_id,
                    "Different entity",
                    f"Different entity {passage_id} held another position.",
                )

        artifact = {
            "question_scope": "Who held the office?",
            "answer_slot": {
                "answer_type": "entity",
                "answer_cardinality": "single",
                "qualifiers": [],
                "proposition_index": 0,
                "answer_field": "subject",
            },
            "evidence_propositions": [
                {
                    "subject": "Unsupported Person",
                    "relation": "held",
                    "object_or_attribute_value": "the office",
                    "qualifiers": [],
                    "evidence_span": "Unsupported Person held the office.",
                }
            ],
            "multi_hop_chain": ["Unsupported Person --held--> the office"],
            "candidate_answer": "Unsupported Person",
            "evidence": ["Unsupported Person held the office."],
        }

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

        queries = (
            "Office holder",
            "Offise holder",
            "Office holder alias",
            "Office holder disambiguation",
            "Who held the office relation",
        )
        outputs: list[str] = []
        for index, (query, limit) in enumerate(
            zip(queries, (5, 10, 15, 20, 25)),
            start=1,
        ):
            outputs.extend(
                [
                    action("search", {"query": query, "limit": limit}),
                    action("read", {"passage_id": f"p{index}"}),
                    action("complete", {"value": artifact}),
                ]
            )
        # After the fifth provenance rejection the public action domain is
        # empty.  This final sampled completion receives the operational
        # coverage diagnosis rather than being admitted as a guess.
        outputs.append(action("complete", {"value": artifact}))

        class SequenceGateway:
            def __init__(self) -> None:
                self.outputs = list(outputs)

            async def generate(self, request: AgentRequest) -> AgentResponse:
                del request
                return AgentResponse(self.outputs.pop(0))

        request = AgentRequest(
            request_id="trivia:coverage-after-reads",
            run_id="trivia",
            graph_revision=1,
            problem="Who held the office?",
            agent=AgentNode(
                "reasoner",
                "model",
                "retrieve evidence and bind the requested entity and relation",
                role_family="reasoner",
                allowed_tools=(QA_RETRIEVAL_TOOL_ID,),
                execution_mode="react",
            ),
            model=ModelSpec("model", "provider"),
            provider=ProviderSpec("provider", kind="test"),
            phase=ExecutionPhase.SINGLE,
            semantic_protocol=QA_VERIFIED_ANSWER_LINEAGE_PROTOCOL,
        )

        with self.assertRaises(ReactExecutionError) as caught:
            await QARetrievalReactExecutionAdapter(
                gateway=SequenceGateway(),
                tool_registry=build_qa_tool_registry(MismatchedIndex()),
                max_turns=16,
                max_tool_calls=10,
                task_type="factual_qa",
                completion_policy="required_evidence",
            ).execute(request)

        error = caught.exception
        self.assertEqual(10, len(error.tool_receipts))
        self.assertIs(True, error.tool_plan_exhausted)
        terminal_diagnosis = error.react_trace[-1][
            "terminal_failure_diagnosis"
        ]
        self.assertIs(True, terminal_diagnosis["tool_plan_exhausted"])
        self.assertEqual(
            "knowledge_base_coverage_failure",
            terminal_diagnosis["public_error_code"],
        )
        self.assertEqual(5, terminal_diagnosis["retrieval_attempt_count"])
        self.assertEqual(
            [5, 10, 15, 20, 25],
            terminal_diagnosis["search_top_ks"],
        )

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
        class EvidenceIndex(FakeIndex):
            def read(self, passage_id: str) -> FakePassage:
                self.read_calls.append(passage_id)
                return FakePassage(
                    passage_id,
                    "d1",
                    "Ada Lovelace",
                    (
                        "Ada Lovelace wrote the algorithm. "
                        "Ada Lovelace was an English mathematician."
                    ),
                )

        index = EvidenceIndex()

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

        semantic_artifact = {
            "question_scope": "Who wrote the first published algorithm?",
            "answer_slot": {
                "answer_type": "entity",
                "answer_cardinality": "single",
                "qualifiers": ["first published algorithm"],
                "proposition_index": 0,
                "answer_field": "subject",
            },
            "evidence_propositions": [
                {
                    "subject": "Ada Lovelace",
                    "relation": "wrote",
                    "object_or_attribute_value": "the first published algorithm",
                    "qualifiers": [],
                    "evidence_span": "Ada Lovelace wrote the algorithm.",
                },
                {
                    "subject": "Ada Lovelace",
                    "relation": "identity",
                    "object_or_attribute_value": "English mathematician",
                    "qualifiers": [],
                    "evidence_span": "Ada Lovelace was an English mathematician.",
                },
            ],
            "multi_hop_chain": ["first published algorithm", "Ada Lovelace"],
            "candidate_answer": "Ada Lovelace",
            "evidence": ["Full public text."],
        }

        class SequenceGateway:
            def __init__(self) -> None:
                self.outputs = [
                    action("complete", {"value": "unsupported answer"}),
                    action("search", {"query": "Ada Lovelace", "limit": 1}),
                    action("read", {"passage_id": "p1"}),
                    action("complete", {"value": semantic_artifact}),
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
                role_family="reasoner",
            ),
            model=ModelSpec("model", "provider"),
            provider=ProviderSpec("provider", kind="test"),
            phase=ExecutionPhase.SINGLE,
            semantic_protocol="hotpotqa_verified_answer_slot_v1",
        )

        response = await QARetrievalReactExecutionAdapter(
            gateway=gateway,
            tool_registry=build_qa_tool_registry(index),
            max_turns=4,
            max_tool_calls=2,
            completion_policy="required_evidence",
        ).execute(request)

        self.assertEqual(
            json.dumps(
                semantic_artifact,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ),
            response.text,
        )
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
            "Search arguments contain exactly query and limit",
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
            "next action must complete",
            gateway.requests[3].agent.contract,
        )
        self.assertIn("Do not use kind=completion", gateway.requests[3].agent.contract)
        search_domain = gateway.requests[1].agent.contract.split(
            "Currently admissible Tool action contracts", 1
        )[1].split("\nCompletion", 1)[0]
        read_domain = gateway.requests[2].agent.contract.split(
            "Currently admissible Tool action contracts", 1
        )[1].split("\nCompletion", 1)[0]
        completion_domain = gateway.requests[3].agent.contract.split(
            "Currently admissible Tool action contracts", 1
        )[1].split("\nCurrently admissible completion", 1)[0]
        self.assertIn('name is "search"', search_domain)
        self.assertNotIn('name is "read"', search_domain)
        self.assertIn("Completion is not admissible", gateway.requests[1].agent.contract)
        self.assertIn('name is "read"', read_domain)
        self.assertNotIn('name is "search"', read_domain)
        self.assertIn("Arguments JSON Schema", search_domain)
        self.assertIn("Arguments JSON Schema", read_domain)
        self.assertNotIn('"argument_json_schema"', search_domain)
        self.assertNotIn('"argument_json_schema"', read_domain)
        self.assertNotIn('"action_envelope"', search_domain)
        self.assertNotIn('"action_envelope"', read_domain)
        self.assertTrue(completion_domain.endswith("- none"))
        self.assertIn(
            "Currently admissible completion schema",
            gateway.requests[3].agent.contract,
        )
        completion_response_schema = json.loads(
            gateway.requests[3].model.metadata["response_json_schema"]
        )
        self.assertEqual(
            ["value"],
            completion_response_schema["properties"]["arguments"]["required"],
        )
        self.assertEqual(
            {
                "question_scope",
                "answer_slot",
                "evidence_propositions",
                "multi_hop_chain",
                "candidate_answer",
                "evidence",
            },
            set(
                completion_response_schema["properties"]["arguments"]
                ["properties"]["value"]["required"]
            ),
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

    async def test_hotpot_multi_hop_reads_two_distinct_search_results_before_complete(
        self,
    ) -> None:
        class TwoPassageIndex(FakeIndex):
            def search(self, query: str, *, limit: int) -> tuple[FakeHit, ...]:
                self.search_calls.append((query, limit))
                return (
                    FakeHit("p1", "d1", "Ada Lovelace", "Ada was a mathematician.", 1),
                    FakeHit("p2", "d2", "Algorithm", "The algorithm was published.", 2),
                    FakeHit("p3", "d3", "Program", "The earliest program.", 3),
                )

            def read(self, passage_id: str) -> FakePassage:
                self.read_calls.append(passage_id)
                text = {
                    "p1": "Ada Lovelace was an English mathematician.",
                    "p2": "Ada Lovelace published the first algorithm.",
                    "p3": "Ada Lovelace authored the earliest computer program.",
                }[passage_id]
                return FakePassage(passage_id, passage_id, passage_id, text)

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

        semantic_artifact = {
            "question_scope": "Who published the first algorithm?",
            "answer_slot": {
                "answer_type": "entity",
                "answer_cardinality": "single",
                "qualifiers": ["first algorithm"],
                "proposition_index": 0,
                "answer_field": "subject",
            },
            "evidence_propositions": [
                {
                    "subject": "Ada Lovelace",
                    "relation": "published",
                    "object_or_attribute_value": "the first algorithm",
                    "qualifiers": [],
                    "evidence_span": "Ada Lovelace published the first algorithm.",
                },
                {
                    "subject": "Ada Lovelace",
                    "relation": "occupation",
                    "object_or_attribute_value": "English mathematician",
                    "qualifiers": [],
                    "evidence_span": "Ada Lovelace was an English mathematician.",
                },
            ],
            "multi_hop_chain": ["Ada Lovelace", "published", "first algorithm"],
            "candidate_answer": "Ada Lovelace",
            "evidence": [
                "Ada Lovelace was an English mathematician.",
                "Ada Lovelace published the first algorithm.",
            ],
        }
        unsupported_span_artifact = json.loads(json.dumps(semantic_artifact))
        unsupported_span_artifact["evidence_propositions"][0][
            "evidence_span"
        ] = "Ada Lovelace authored the earliest computer program."

        class SequenceGateway:
            def __init__(self) -> None:
                self.outputs = [
                    action("search", {"query": "Ada Lovelace algorithm", "limit": 10}),
                    action("read", {"passage_id": "p1"}),
                    action("search", {"query": "first published algorithm", "limit": 10}),
                    action("read", {"passage_id": "p2"}),
                    action(
                        "complete",
                        {"value": unsupported_span_artifact},
                    ),
                    action("search", {"query": "earliest computer program", "limit": 10}),
                    action("read", {"passage_id": "p3"}),
                    action("complete", {"value": unsupported_span_artifact}),
                ]
                self.requests: list[AgentRequest] = []

            async def generate(self, request: AgentRequest) -> AgentResponse:
                self.requests.append(request)
                return AgentResponse(self.outputs.pop(0))

        index = TwoPassageIndex()
        gateway = SequenceGateway()
        request = AgentRequest(
            request_id="qa:two-hop",
            run_id="qa",
            graph_revision=1,
            problem="Who published the first algorithm?",
            agent=AgentNode(
                "reasoner",
                "model",
                "retrieve both supporting passages and determine the answer",
                role_family="reasoner",
                allowed_tools=(QA_RETRIEVAL_TOOL_ID,),
                execution_mode="react",
            ),
            model=ModelSpec("model", "provider"),
            provider=ProviderSpec("provider", kind="test"),
            phase=ExecutionPhase.SINGLE,
            semantic_protocol="hotpotqa_verified_answer_slot_v1",
        )

        response = await QARetrievalReactExecutionAdapter(
            gateway=gateway,
            tool_registry=build_qa_tool_registry(index),
            max_turns=8,
            max_tool_calls=6,
            task_type="multi_hop_qa",
            completion_policy="required_evidence",
        ).execute(request)

        self.assertEqual(json.dumps(unsupported_span_artifact, ensure_ascii=False, sort_keys=True, separators=(",", ":")), response.text)
        self.assertEqual(6, response.metadata["tool_calls"])
        self.assertEqual(["p1", "p2", "p3"], index.read_calls)
        self.assertEqual(3, len(index.search_calls))
        self.assertTrue(
            response.metadata["react_trace"][4]["public_error_code"].startswith(
                "hotpotqa_semantic_evidence_provenance_invalid:"
            )
        )
        self.assertIn(
            "has no typography-canonical lexical match",
            response.metadata["react_trace"][4]["public_error_code"],
        )
        search_schema = json.loads(
            gateway.requests[0].model.metadata["response_json_schema"]
        )
        self.assertEqual(
            10,
            search_schema["properties"]["arguments"]["properties"]["limit"]["const"],
        )
        second_read_schema = json.loads(
            gateway.requests[3].model.metadata["response_json_schema"]
        )
        self.assertIn(
            "p2",
            second_read_schema["properties"]["arguments"]["properties"]
            ["passage_id"]["enum"],
        )
        self.assertIn("next action must be qa-retrieval search", gateway.requests[2].agent.contract)
        self.assertIn("next action must be qa-retrieval read", gateway.requests[3].agent.contract)
        self.assertIn("next action must complete", gateway.requests[4].agent.contract)
        self.assertIn("next action must be qa-retrieval search", gateway.requests[5].agent.contract)
        self.assertIn(
            "use the admitted qa-retrieval search/read continuation",
            gateway.requests[5].agent.contract,
        )
        self.assertIn("next action must be qa-retrieval read", gateway.requests[6].agent.contract)
        completion_schema = json.loads(
            gateway.requests[4].model.metadata["response_json_schema"]
        )
        semantic_properties = completion_schema["properties"]["arguments"][
            "properties"
        ]["value"]["properties"]
        self.assertEqual(
            "entity",
            semantic_properties["answer_slot"]["properties"]["answer_type"][
                "const"
            ],
        )
        self.assertEqual(
            0,
            semantic_properties["answer_slot"]["properties"][
                "proposition_index"
            ]["minimum"],
        )
        self.assertEqual(
            "single",
            semantic_properties["answer_slot"]["properties"][
                "answer_cardinality"
            ]["const"],
        )
        self.assertEqual(
            {
                "answer_type",
                "answer_cardinality",
                "qualifiers",
                "proposition_index",
                "answer_field",
            },
            set(semantic_properties["answer_slot"]["required"]),
        )

    async def test_hotpot_reasoner_accepts_direct_retriever_read_provenance_without_spending_own_tool_budget(
        self,
    ) -> None:
        evidence_text = (
            "Ada Lovelace was an English mathematician. "
            "Ada Lovelace published the first algorithm."
        )
        upstream_read_receipt = {
            "tool_id": QA_RETRIEVAL_TOOL_ID,
            "tool_version": "frozen-index-v1",
            "request": {
                "action": "read",
                "arguments": {"passage_id": "p1"},
            },
            "result": {
                "value": {
                    "operation": "read",
                    "passage_id": "p1",
                    "passage": {
                        "passage_id": "p1",
                        "text": evidence_text,
                    },
                },
                "completed": True,
            },
            "error_type": None,
        }
        artifact = {
            "question_scope": "Who published the first algorithm?",
            "answer_slot": {
                "answer_type": "entity",
                "answer_cardinality": "single",
                "qualifiers": ["first algorithm"],
                "proposition_index": 0,
                "answer_field": "subject",
            },
            "evidence_propositions": [
                {
                    "subject": "Ada Lovelace",
                    "relation": "published",
                    "object_or_attribute_value": "the first algorithm",
                    "qualifiers": [],
                    "evidence_span": (
                        "Ada Lovelace published the first algorithm."
                    ),
                },
                {
                    "subject": "Ada Lovelace",
                    "relation": "occupation",
                    "object_or_attribute_value": "English mathematician",
                    "qualifiers": [],
                    "evidence_span": (
                        "Ada Lovelace was an English mathematician."
                    ),
                },
            ],
            "multi_hop_chain": [
                "Ada Lovelace --occupation--> English mathematician",
                "Ada Lovelace --published--> the first algorithm",
            ],
            "candidate_answer": "Ada Lovelace",
            "evidence": [evidence_text],
        }

        class CompletionGateway:
            async def generate(self, request: AgentRequest) -> AgentResponse:
                del request
                return AgentResponse(
                    json.dumps(
                        {
                            "arguments": {"value": artifact},
                            "kind": "complete",
                            "name": "complete",
                            "resource_id": None,
                            "skill_id": None,
                        }
                    )
                )

        request = AgentRequest(
            request_id="qa:upstream-retrieval-provenance",
            run_id="qa",
            graph_revision=1,
            problem="Who published the first algorithm?",
            agent=AgentNode(
                "reasoner",
                "model",
                "bind direct predecessor evidence to the answer slot",
                role_family="reasoner",
                allowed_tools=(QA_RETRIEVAL_TOOL_ID,),
                execution_mode="react",
            ),
            model=ModelSpec("model", "provider"),
            provider=ProviderSpec("provider", kind="test"),
            phase=ExecutionPhase.SINGLE,
            semantic_protocol="hotpotqa_verified_answer_slot_v1",
            upstream=(
                UpstreamMessage(
                    "evidence_retriever",
                    "reasoner",
                    "retrieved passage p1",
                    graph_revision=1,
                    artifact_type="retrieval_evidence",
                    tool_receipts=(upstream_read_receipt,),
                ),
            ),
        )

        response = await QARetrievalReactExecutionAdapter(
            gateway=CompletionGateway(),
            tool_registry=build_qa_tool_registry(FakeIndex()),
            max_turns=1,
            max_tool_calls=2,
            task_type="multi_hop_qa",
            completion_policy="required_evidence",
        ).execute(request)

        self.assertEqual(
            json.dumps(
                artifact,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ),
            response.text,
        )
        self.assertEqual(0, response.metadata["tool_calls"])
        self.assertEqual(0, response.metadata["continued_tool_receipt_count"])
        self.assertEqual([], response.metadata["tool_receipts"])

    async def test_hotpot_reasoner_rejects_forged_span_despite_upstream_read_receipt(
        self,
    ) -> None:
        upstream_read_receipt = {
            "tool_id": QA_RETRIEVAL_TOOL_ID,
            "tool_version": "frozen-index-v1",
            "request": {
                "action": "read",
                "arguments": {"passage_id": "p1"},
            },
            "result": {
                "value": {
                    "operation": "read",
                    "passage_id": "p1",
                    "passage": {
                        "passage_id": "p1",
                        "text": (
                            "Ada Lovelace was an English mathematician. "
                            "Ada Lovelace published the first algorithm."
                        ),
                    },
                },
                "completed": True,
            },
            "error_type": None,
        }
        artifact = {
            "question_scope": "Who published the first algorithm?",
            "answer_slot": {
                "answer_type": "entity",
                "answer_cardinality": "single",
                "qualifiers": ["first algorithm"],
                "proposition_index": 0,
                "answer_field": "subject",
            },
            "evidence_propositions": [
                {
                    "subject": "Ada Lovelace",
                    "relation": "published",
                    "object_or_attribute_value": "the first algorithm",
                    "qualifiers": [],
                    "evidence_span": "A forged passage states the answer.",
                },
                {
                    "subject": "Ada Lovelace",
                    "relation": "occupation",
                    "object_or_attribute_value": "English mathematician",
                    "qualifiers": [],
                    "evidence_span": (
                        "Ada Lovelace was an English mathematician."
                    ),
                },
            ],
            "multi_hop_chain": [
                "Ada Lovelace --occupation--> English mathematician",
                "Ada Lovelace --published--> the first algorithm",
            ],
            "candidate_answer": "Ada Lovelace",
            "evidence": ["A forged passage states the answer."],
        }

        class CompletionGateway:
            async def generate(self, request: AgentRequest) -> AgentResponse:
                del request
                return AgentResponse(
                    json.dumps(
                        {
                            "arguments": {"value": artifact},
                            "kind": "complete",
                            "name": "complete",
                            "resource_id": None,
                            "skill_id": None,
                        }
                    )
                )

        request = AgentRequest(
            request_id="qa:forged-upstream-retrieval-provenance",
            run_id="qa",
            graph_revision=1,
            problem="Who published the first algorithm?",
            agent=AgentNode(
                "reasoner",
                "model",
                "bind direct predecessor evidence to the answer slot",
                role_family="reasoner",
                allowed_tools=(QA_RETRIEVAL_TOOL_ID,),
                execution_mode="react",
            ),
            model=ModelSpec("model", "provider"),
            provider=ProviderSpec("provider", kind="test"),
            phase=ExecutionPhase.SINGLE,
            semantic_protocol="hotpotqa_verified_answer_slot_v1",
            upstream=(
                UpstreamMessage(
                    "evidence_retriever",
                    "reasoner",
                    "retrieved passage p1",
                    graph_revision=1,
                    artifact_type="retrieval_evidence",
                    tool_receipts=(upstream_read_receipt,),
                ),
            ),
        )

        with self.assertRaises(ReactExecutionError) as caught:
            await QARetrievalReactExecutionAdapter(
                gateway=CompletionGateway(),
                tool_registry=build_qa_tool_registry(FakeIndex()),
                max_turns=1,
                max_tool_calls=2,
                task_type="multi_hop_qa",
                completion_policy="required_evidence",
            ).execute(request)

        self.assertTrue(
            caught.exception.react_trace[-1]["public_error_code"].startswith(
                "hotpotqa_semantic_evidence_provenance_invalid:"
            )
        )
        self.assertEqual(0, len(caught.exception.tool_receipts))

    async def test_hotpot_provenance_rejection_marks_tool_plan_exhausted_when_pair_cannot_fit(
        self,
    ) -> None:
        upstream_read_receipt = {
            "tool_id": QA_RETRIEVAL_TOOL_ID,
            "tool_version": "frozen-index-v1",
            "request": {
                "action": "read",
                "arguments": {"passage_id": "p1"},
            },
            "result": {
                "value": {
                    "operation": "read",
                    "passage_id": "p1",
                    "passage": {
                        "passage_id": "p1",
                        "text": (
                            "Ada Lovelace was an English mathematician. "
                            "Ada Lovelace published the first algorithm."
                        ),
                    },
                },
                "completed": True,
            },
            "error_type": None,
        }
        artifact = {
            "question_scope": "Who published the first algorithm?",
            "answer_slot": {
                "answer_type": "entity",
                "answer_cardinality": "single",
                "qualifiers": ["first algorithm"],
                "proposition_index": 0,
                "answer_field": "subject",
            },
            "evidence_propositions": [
                {
                    "subject": "Ada Lovelace",
                    "relation": "published",
                    "object_or_attribute_value": "the first algorithm",
                    "qualifiers": [],
                    "evidence_span": "A forged passage states the answer.",
                },
                {
                    "subject": "Ada Lovelace",
                    "relation": "occupation",
                    "object_or_attribute_value": "English mathematician",
                    "qualifiers": [],
                    "evidence_span": (
                        "Ada Lovelace was an English mathematician."
                    ),
                },
            ],
            "multi_hop_chain": [
                "Ada Lovelace --occupation--> English mathematician",
                "Ada Lovelace --published--> the first algorithm",
            ],
            "candidate_answer": "Ada Lovelace",
            "evidence": ["A forged passage states the answer."],
        }

        class CompletionGateway:
            async def generate(self, request: AgentRequest) -> AgentResponse:
                del request
                return AgentResponse(
                    json.dumps(
                        {
                            "arguments": {"value": artifact},
                            "kind": "complete",
                            "name": "complete",
                            "resource_id": None,
                            "skill_id": None,
                        }
                    )
                )

        request = AgentRequest(
            request_id="qa:upstream-provenance-tool-plan-exhausted",
            run_id="qa",
            graph_revision=1,
            problem="Who published the first algorithm?",
            agent=AgentNode(
                "reasoner",
                "model",
                "bind direct predecessor evidence to the answer slot",
                role_family="reasoner",
                allowed_tools=(QA_RETRIEVAL_TOOL_ID,),
                execution_mode="react",
            ),
            model=ModelSpec("model", "provider"),
            provider=ProviderSpec("provider", kind="test"),
            phase=ExecutionPhase.SINGLE,
            semantic_protocol="hotpotqa_verified_answer_slot_v1",
            upstream=(
                UpstreamMessage(
                    "evidence_retriever",
                    "reasoner",
                    "retrieved passage p1",
                    graph_revision=1,
                    artifact_type="retrieval_evidence",
                    tool_receipts=(upstream_read_receipt,),
                ),
            ),
        )

        with self.assertRaises(ReactExecutionError) as caught:
            await QARetrievalReactExecutionAdapter(
                gateway=CompletionGateway(),
                tool_registry=build_qa_tool_registry(FakeIndex()),
                max_turns=1,
                max_tool_calls=1,
                task_type="multi_hop_qa",
                completion_policy="required_evidence",
            ).execute(request)

        self.assertIs(True, caught.exception.tool_plan_exhausted)
        terminal_diagnosis = caught.exception.react_trace[-1][
            "terminal_failure_diagnosis"
        ]
        self.assertIs(True, terminal_diagnosis["tool_plan_exhausted"])
        self.assertEqual(1, terminal_diagnosis["remaining_tool_calls"])
        self.assertEqual(0, len(caught.exception.tool_receipts))

    async def test_structural_semantic_rejection_repairs_with_existing_evidence(
        self,
    ) -> None:
        class TwoPassageIndex(FakeIndex):
            def search(self, query: str, *, limit: int) -> tuple[FakeHit, ...]:
                self.search_calls.append((query, limit))
                return (
                    FakeHit("p1", "d1", "Ada Lovelace", "Occupation.", 1),
                    FakeHit("p2", "d2", "Algorithm", "Publication.", 2),
                )

            def read(self, passage_id: str) -> FakePassage:
                self.read_calls.append(passage_id)
                text = {
                    "p1": "Ada Lovelace was an English mathematician.",
                    "p2": "Ada Lovelace published the first algorithm.",
                }[passage_id]
                return FakePassage(passage_id, passage_id, passage_id, text)

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

        valid_artifact = {
            "question_scope": "Who published the first algorithm?",
            "answer_slot": {
                "answer_type": "entity",
                "answer_cardinality": "single",
                "qualifiers": ["first algorithm"],
                "proposition_index": 0,
                "answer_field": "subject",
            },
            "evidence_propositions": [
                {
                    "subject": "Ada Lovelace",
                    "relation": "published",
                    "object_or_attribute_value": "the first algorithm",
                    "qualifiers": [],
                    "evidence_span": "Ada Lovelace published the first algorithm.",
                },
                {
                    "subject": "Ada Lovelace",
                    "relation": "occupation",
                    "object_or_attribute_value": "English mathematician",
                    "qualifiers": [],
                    "evidence_span": "Ada Lovelace was an English mathematician.",
                },
            ],
            "multi_hop_chain": ["Ada Lovelace", "published", "first algorithm"],
            "candidate_answer": "Ada Lovelace",
            "evidence": [
                "Ada Lovelace was an English mathematician.",
                "Ada Lovelace published the first algorithm.",
            ],
        }
        binding_mismatch = json.loads(json.dumps(valid_artifact))
        binding_mismatch["candidate_answer"] = "the first algorithm"

        class SequenceGateway:
            def __init__(self) -> None:
                self.outputs = [
                    action("search", {"query": "Ada Lovelace", "limit": 10}),
                    action("read", {"passage_id": "p1"}),
                    action("search", {"query": "first algorithm", "limit": 10}),
                    action("read", {"passage_id": "p2"}),
                    action("complete", {"value": binding_mismatch}),
                    action("complete", {"value": valid_artifact}),
                ]
                self.requests: list[AgentRequest] = []

            async def generate(self, request: AgentRequest) -> AgentResponse:
                self.requests.append(request)
                return AgentResponse(self.outputs.pop(0))

        index = TwoPassageIndex()
        gateway = SequenceGateway()
        request = AgentRequest(
            request_id="qa:structural-semantic-repair",
            run_id="qa",
            graph_revision=1,
            problem="Who published the first algorithm?",
            agent=AgentNode(
                "reasoner",
                "model",
                "retrieve both supporting passages and determine the answer",
                role_family="reasoner",
                allowed_tools=(QA_RETRIEVAL_TOOL_ID,),
                execution_mode="react",
            ),
            model=ModelSpec("model", "provider"),
            provider=ProviderSpec("provider", kind="test"),
            phase=ExecutionPhase.SINGLE,
            semantic_protocol="hotpotqa_verified_answer_slot_v1",
        )

        response = await QARetrievalReactExecutionAdapter(
            gateway=gateway,
            tool_registry=build_qa_tool_registry(index),
            max_turns=6,
            max_tool_calls=4,
            task_type="multi_hop_qa",
            completion_policy="required_evidence",
        ).execute(request)

        self.assertEqual(
            json.dumps(
                valid_artifact,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ),
            response.text,
        )
        self.assertEqual(4, response.metadata["tool_calls"])
        self.assertEqual(2, len(index.search_calls))
        self.assertEqual(["p1", "p2"], index.read_calls)
        structural_error = response.metadata["react_trace"][4][
            "public_error_code"
        ]
        self.assertTrue(
            structural_error.startswith("hotpotqa_semantic_artifact_invalid:")
        )
        self.assertIn("answer_slot.answer_field selects", structural_error)
        self.assertIn(
            "object_or_attribute_value",
            structural_error,
        )

        repair_contract = gateway.requests[5].agent.contract
        self.assertIn(structural_error, repair_contract)
        self.assertIn(
            "Repair only the diagnosed structured semantic artifact fields",
            repair_contract,
        )
        self.assertIn(
            "Change only answer_slot.answer_field from 'subject' to "
            "'object_or_attribute_value'",
            repair_contract,
        )
        self.assertIn(
            "minimal but complete evidence-aligned referential surface",
            repair_contract,
        )
        self.assertIn("complete possessor entity mention", repair_contract)
        self.assertIn("possessive marker", repair_contract)
        self.assertIn("do not add a search or read", repair_contract)
        repair_domain = repair_contract.split(
            "Currently admissible Tool action contracts", 1
        )[1].split("\nCurrently admissible completion", 1)[0]
        self.assertTrue(repair_domain.endswith("- none"))
        self.assertIn("Currently admissible completion schema", repair_contract)

    async def test_unified_entity_binding_repairs_on_preserved_read_evidence(
        self,
    ) -> None:
        class DenchIndex(FakeIndex):
            def search(self, query: str, *, limit: int) -> tuple[FakeHit, ...]:
                self.search_calls.append((query, limit))
                return (
                    FakeHit(
                        "p1",
                        "d1",
                        "Judi Dench",
                        "Dench was born in Heworth.",
                        1,
                    ),
                )

            def read(self, passage_id: str) -> FakePassage:
                self.read_calls.append(passage_id)
                return FakePassage(
                    passage_id,
                    "d1",
                    "Judi Dench",
                    "Dench was born in Heworth, North Riding of Yorkshire.",
                )

        question = "Where in England was Dame Judi Dench born?"
        valid_artifact = {
            "question_scope": question,
            "answer_slot": {
                "answer_type": "location",
                "answer_cardinality": "single",
                "qualifiers": ["in England"],
                "proposition_index": 0,
                "answer_field": "object_or_attribute_value",
            },
            "evidence_propositions": [
                {
                    "subject": "Dench",
                    "relation": "was born in",
                    "object_or_attribute_value": (
                        "Heworth, North Riding of Yorkshire"
                    ),
                    "qualifiers": ["in England"],
                    "evidence_span": (
                        "Dench was born in Heworth, North Riding of Yorkshire."
                    ),
                }
            ],
            "multi_hop_chain": [
                "Dench --was born in--> Heworth, North Riding of Yorkshire"
            ],
            "candidate_answer": "Heworth, North Riding of Yorkshire",
            "evidence": ["p1"],
        }
        ungrounded_entity_surface = json.loads(json.dumps(valid_artifact))
        ungrounded_entity_surface["evidence_propositions"][0]["subject"] = (
            "Dame Judi Dench"
        )

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
                    action(
                        "search",
                        {"query": "Dame Judi Dench birthplace", "limit": 5},
                    ),
                    action("read", {"passage_id": "p1"}),
                    action("complete", {"value": ungrounded_entity_surface}),
                    action("complete", {"value": valid_artifact}),
                ]
                self.requests: list[AgentRequest] = []

            async def generate(self, request: AgentRequest) -> AgentResponse:
                self.requests.append(request)
                return AgentResponse(self.outputs.pop(0))

        index = DenchIndex()
        gateway = SequenceGateway()
        request = AgentRequest(
            request_id="trivia:entity-binding-repair",
            run_id="trivia",
            graph_revision=1,
            problem=question,
            agent=AgentNode(
                "reasoner",
                "model",
                "retrieve evidence and preserve the requested relation",
                role_family="reasoner",
                allowed_tools=(QA_RETRIEVAL_TOOL_ID,),
                execution_mode="react",
            ),
            model=ModelSpec("model", "provider"),
            provider=ProviderSpec("provider", kind="test"),
            phase=ExecutionPhase.SINGLE,
            semantic_protocol=QA_VERIFIED_ANSWER_LINEAGE_PROTOCOL,
        )

        response = await QARetrievalReactExecutionAdapter(
            gateway=gateway,
            tool_registry=build_qa_tool_registry(index),
            max_turns=4,
            max_tool_calls=10,
            task_type="factual_qa",
            completion_policy="required_evidence",
        ).execute(request)

        self.assertEqual([("Dame Judi Dench birthplace", 5)], index.search_calls)
        self.assertEqual(["p1"], index.read_calls)
        public_error = response.metadata["react_trace"][2]["public_error_code"]
        self.assertTrue(public_error.startswith("qa_semantic_artifact_invalid:"))
        self.assertIn("no deterministic entity binding", public_error)
        repair_contract = gateway.requests[3].agent.contract
        self.assertIn("do not add a search or read", repair_contract)

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

    def test_evidence_retriever_schema_and_contract_are_answer_free(self) -> None:
        adapter = QARetrievalReactExecutionAdapter(
            gateway=SimpleNamespace(generate=lambda request: None),
            tool_registry=build_qa_tool_registry(FakeIndex()),
            max_turns=3,
            max_tool_calls=2,
            task_type="factual_qa",
            completion_policy="required_evidence",
        )
        request = AgentRequest(
            request_id="trivia:evidence-retriever-schema",
            run_id="trivia",
            graph_revision=1,
            problem="Which city does David Soul come from?",
            agent=AgentNode(
                "evidence_retriever",
                "model",
                "retrieve evidence for the requested entity and relation",
                role_family="evidence_retriever",
                allowed_tools=(QA_RETRIEVAL_TOOL_ID,),
                execution_mode="react",
            ),
            model=ModelSpec("model", "provider"),
            provider=ProviderSpec("provider", kind="test"),
            phase=ExecutionPhase.SINGLE,
            semantic_protocol=QA_VERIFIED_ANSWER_LINEAGE_PROTOCOL,
        )

        schema = adapter._completion_arguments_schema(request)
        artifact_schema = schema["properties"]["value"]
        expected_fields = {
            "question_scope",
            "entity_identity",
            "target_relation",
            "answer_type_constraint",
            "evidence_proposition",
            "evidence_span",
            "passage_id",
        }
        self.assertEqual(expected_fields, set(artifact_schema["required"]))
        self.assertEqual(expected_fields, set(artifact_schema["properties"]))
        self.assertFalse(artifact_schema["additionalProperties"])
        self.assertEqual(
            {"question_surface", "evidence_surface"},
            set(
                artifact_schema["properties"]["entity_identity"]["required"]
            ),
        )
        for answer_field in ("candidate_answer", "answer_slot", "final_answer"):
            self.assertNotIn(answer_field, artifact_schema["properties"])
        self.assertIn(
            "question-side entity or event anchor",
            artifact_schema["properties"]["entity_identity"]["properties"][
                "question_surface"
            ]["description"],
        )
        self.assertIn(
            "must not be a wh-word or wh-phrase",
            artifact_schema["properties"]["entity_identity"]["properties"][
                "question_surface"
            ]["description"],
        )
        self.assertIn(
            "coreferential surface of the same",
            artifact_schema["properties"]["entity_identity"]["properties"][
                "evidence_surface"
            ]["description"],
        )
        self.assertIn(
            "exact predicate surface",
            artifact_schema["properties"]["target_relation"]["description"],
        )
        self.assertEqual(
            "entity",
            artifact_schema["properties"]["answer_type_constraint"]["const"],
        )
        proposition_schema = artifact_schema["properties"]["evidence_proposition"]
        self.assertEqual(
            {"subject", "predicate", "object_or_attribute_value"},
            set(proposition_schema["required"]),
        )

        completion_contract = adapter._contract(
            request,
            [
                {
                    "observation_status": "success",
                    "result": {
                        "operation": "read",
                        "passage_id": "p1",
                        "passage": {
                            "passage_id": "p1",
                            "text": "David Soul is from Chicago.",
                        },
                    },
                }
            ],
        )
        self.assertIn("As the Evidence Retriever", completion_contract)
        self.assertIn(
            "Cite one successful qa-retrieval read receipt",
            completion_contract,
        )
        self.assertIn(
            "Do not select or emit candidate_answer, answer_slot, or final_answer",
            completion_contract,
        )
        self.assertIn("question-side entity/event anchor", completion_contract)
        self.assertIn("never a wh-word/wh-phrase", completion_contract)
        self.assertIn("passage title for explicit alias binding", completion_contract)
        self.assertIn(
            "question_surface and evidence_surface differ",
            completion_contract,
        )
        self.assertIn(
            "evidence_surface must be a coreferential identity surface",
            completion_contract,
        )
        self.assertIn(
            "may occupy the subject, the object, or neither",
            completion_contract,
        )

    def test_role_conditional_retriever_and_verifier_do_not_require_reasoner(
        self,
    ) -> None:
        adapter = QARetrievalReactExecutionAdapter(
            gateway=SimpleNamespace(generate=lambda request: None),
            tool_registry=build_qa_tool_registry(FakeIndex()),
            max_turns=5,
            max_tool_calls=4,
            task_type="multi_hop_qa",
            completion_policy="required_evidence",
        )
        request = AgentRequest(
            request_id="hotpot:open-retriever",
            run_id="hotpot",
            graph_revision=1,
            problem="Which target is reached through Bridge Beta?",
            agent=AgentNode(
                "retriever",
                "model",
                "retrieve receipt-grounded evidence",
                role_family="evidence_retriever",
                allowed_tools=(QA_RETRIEVAL_TOOL_ID,),
                execution_mode="react",
            ),
            model=ModelSpec("model", "provider"),
            provider=ProviderSpec("provider", kind="test"),
            phase=ExecutionPhase.SINGLE,
            semantic_protocol="hotpotqa_role_conditional_capabilities_v1",
        )
        observations = [
            {
                "observation_status": "success",
                "result": {
                    "operation": "read",
                    "passage_id": passage_id,
                    "passage": {"text": f"Evidence from {passage_id}."},
                },
            }
            for passage_id in ("p1", "p2")
        ]

        retriever_contract = adapter._contract(request, observations)
        self.assertIn(
            "a downstream semantic producer owns semantic alignment",
            retriever_contract,
        )
        self.assertNotIn("the Reasoner owns semantic alignment", retriever_contract)

        verifier_contract = adapter._contract(
            replace(
                request,
                request_id="hotpot:open-verifier",
                agent=replace(
                    request.agent,
                    id="verifier",
                    role_family="verifier",
                    contract="verify the routed semantic candidate",
                ),
            ),
            observations,
        )
        self.assertIn(
            "Do not replace the routed semantic producer's candidate",
            verifier_contract,
        )
        self.assertNotIn("Reasoner's candidate", verifier_contract)

    def test_evidence_retriever_accepts_david_soul_title_alias_binding(
        self,
    ) -> None:
        question = "Which city does David Soul come from?"
        evidence = "Soul was born in Chicago, Illinois."
        artifact = {
            "question_scope": question,
            "entity_identity": {
                "question_surface": "David Soul",
                "evidence_surface": "Soul",
            },
            "target_relation": "was born in",
            "answer_type_constraint": "entity",
            "evidence_proposition": {
                "subject": "Soul",
                "predicate": "was born in",
                "object_or_attribute_value": "Chicago, Illinois",
            },
            "evidence_span": evidence,
            "passage_id": "david-soul",
        }
        receipt = {
            "tool_id": QA_RETRIEVAL_TOOL_ID,
            "tool_version": "frozen-index-v1",
            "request": {
                "action": "read",
                "arguments": {"passage_id": "david-soul"},
            },
            "result": {
                "value": {
                    "operation": "read",
                    "passage_id": "david-soul",
                    "passage": {
                        "passage_id": "david-soul",
                        "title": "David Soul",
                        "text": evidence,
                    },
                },
                "completed": True,
            },
            "error_type": None,
        }

        issue = QARetrievalReactExecutionAdapter._evidence_retriever_completion_issue
        self.assertIsNone(
            issue(
                original_question=question,
                artifact=json.dumps(artifact),
                tool_receipts=[receipt],
            )
        )

        wh_anchor = json.loads(json.dumps(artifact))
        wh_anchor["entity_identity"]["question_surface"] = "Which city"
        detail = issue(
            original_question=question,
            artifact=json.dumps(wh_anchor),
            tool_receipts=[receipt],
        )
        self.assertIsNotNone(detail)
        assert detail is not None
        self.assertIn("not a wh-word or wh-phrase", detail)

    def test_factual_evidence_retriever_honorific_title_alias_chain_is_exact(
        self,
    ) -> None:
        question = "Where in England was Dame Judi Dench born?"
        evidence = "Dench was born in Heworth, North Riding of Yorkshire."
        artifact = {
            "question_scope": question,
            "entity_identity": {
                "question_surface": "Dame Judi Dench",
                "evidence_surface": "Dench",
            },
            "target_relation": "was born in",
            "answer_type_constraint": "location",
            "evidence_proposition": {
                "subject": "Dench",
                "predicate": "was born in",
                "object_or_attribute_value": (
                    "Heworth, North Riding of Yorkshire"
                ),
            },
            "evidence_span": evidence,
            "passage_id": "dench-birthplace",
        }

        def receipt(title: str | None) -> dict[str, object]:
            passage = {
                "passage_id": "dench-birthplace",
                "text": evidence,
            }
            if title is not None:
                passage["title"] = title
            return {
                "tool_id": QA_RETRIEVAL_TOOL_ID,
                "tool_version": "frozen-index-v1",
                "request": {
                    "action": "read",
                    "arguments": {"passage_id": "dench-birthplace"},
                },
                "result": {
                    "value": {
                        "operation": "read",
                        "passage_id": "dench-birthplace",
                        "passage": passage,
                    },
                    "completed": True,
                },
                "error_type": None,
            }

        issue = QARetrievalReactExecutionAdapter._evidence_retriever_completion_issue
        self.assertIsNone(
            issue(
                original_question=question,
                artifact=json.dumps(artifact),
                tool_receipts=[receipt("Judi Dench")],
            )
        )

        for title in (None, "Judy Dench"):
            with self.subTest(title=title):
                detail = issue(
                    original_question=question,
                    artifact=json.dumps(artifact),
                    tool_receipts=[receipt(title)],
                )
                self.assertIsNotNone(detail)
                assert detail is not None
                self.assertIn("alias identity lacks an explicit binding", detail)

        answer_in_identity = json.loads(json.dumps(artifact))
        answer_in_identity["entity_identity"]["evidence_surface"] = (
            "Heworth, North Riding of Yorkshire"
        )
        detail = issue(
            original_question=question,
            artifact=json.dumps(answer_in_identity),
            tool_receipts=[receipt("Judi Dench")],
        )
        self.assertIsNotNone(detail)
        assert detail is not None
        self.assertIn("passage title identity chain", detail)

    def test_title_bound_identity_may_bind_reverse_relation_object(self) -> None:
        question = "Who was represented by Dr Alice Carter?"
        evidence = "Morgan Reed was represented by Carter."
        artifact = {
            "question_scope": question,
            "entity_identity": {
                "question_surface": "Dr Alice Carter",
                "evidence_surface": "Carter",
            },
            "target_relation": "was represented by",
            "answer_type_constraint": "entity",
            "evidence_proposition": {
                "subject": "Morgan Reed",
                "predicate": "was represented by",
                "object_or_attribute_value": "Carter",
            },
            "evidence_span": evidence,
            "passage_id": "alice-carter",
        }
        receipt = {
            "tool_id": QA_RETRIEVAL_TOOL_ID,
            "tool_version": "frozen-index-v1",
            "request": {
                "action": "read",
                "arguments": {"passage_id": "alice-carter"},
            },
            "result": {
                "value": {
                    "operation": "read",
                    "passage_id": "alice-carter",
                    "passage": {
                        "passage_id": "alice-carter",
                        "title": "Alice Carter",
                        "text": evidence,
                    },
                },
                "completed": True,
            },
            "error_type": None,
        }

        self.assertIsNone(
            QARetrievalReactExecutionAdapter._evidence_retriever_completion_issue(
                original_question=question,
                artifact=json.dumps(artifact),
                tool_receipts=[receipt],
            )
        )

    async def test_factual_identity_repair_stays_completion_only_at_tool_budget(
        self,
    ) -> None:
        question = "Where in England was Dame Judi Dench born?"
        evidence = "Dench was born in Heworth, North Riding of Yorkshire."
        valid_artifact = {
            "question_scope": question,
            "entity_identity": {
                "question_surface": "Dame Judi Dench",
                "evidence_surface": "Dench",
            },
            "target_relation": "was born in",
            "answer_type_constraint": "location",
            "evidence_proposition": {
                "subject": "Dench",
                "predicate": "was born in",
                "object_or_attribute_value": (
                    "Heworth, North Riding of Yorkshire"
                ),
            },
            "evidence_span": evidence,
            "passage_id": "p1",
        }
        invalid_artifact = json.loads(json.dumps(valid_artifact))
        invalid_artifact["entity_identity"]["evidence_surface"] = (
            "Heworth, North Riding of Yorkshire"
        )

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

        class DenchIndex(FakeIndex):
            def read(self, passage_id: str) -> FakePassage:
                self.read_calls.append(passage_id)
                return FakePassage(passage_id, "d1", "Judi Dench", evidence)

        class SequenceGateway:
            def __init__(self) -> None:
                self.requests: list[AgentRequest] = []
                self.outputs = [
                    action(
                        "search",
                        {"query": "Dame Judi Dench birthplace", "limit": 5},
                    ),
                    action("read", {"passage_id": "p1"}),
                    action("complete", {"value": invalid_artifact}),
                    action("complete", {"value": valid_artifact}),
                ]

            async def generate(self, request: AgentRequest) -> AgentResponse:
                self.requests.append(request)
                return AgentResponse(self.outputs.pop(0))

        index = DenchIndex()
        gateway = SequenceGateway()
        response = await QARetrievalReactExecutionAdapter(
            gateway=gateway,
            tool_registry=build_qa_tool_registry(index),
            max_turns=4,
            max_tool_calls=2,
            task_type="factual_qa",
            completion_policy="required_evidence",
        ).execute(
            AgentRequest(
                request_id="trivia:honorific-identity-repair",
                run_id="trivia",
                graph_revision=1,
                problem=question,
                agent=AgentNode(
                    "evidence_retriever",
                    "model",
                    "retrieve receipt-grounded identity and birthplace evidence",
                    role_family="evidence_retriever",
                    allowed_tools=(QA_RETRIEVAL_TOOL_ID,),
                    execution_mode="react",
                ),
                model=ModelSpec("model", "provider"),
                provider=ProviderSpec("provider", kind="test"),
                phase=ExecutionPhase.SINGLE,
                semantic_protocol=QA_VERIFIED_ANSWER_LINEAGE_PROTOCOL,
            )
        )

        self.assertEqual([("Dame Judi Dench birthplace", 5)], index.search_calls)
        self.assertEqual(["p1"], index.read_calls)
        self.assertEqual(2, response.metadata["tool_calls"])
        feedback = response.metadata["react_trace"][2]["public_error_code"]
        self.assertTrue(feedback.startswith("qa_semantic_artifact_invalid:"))
        self.assertNotIn("knowledge_base_coverage_failure", feedback)
        repair_contract = gateway.requests[3].agent.contract
        self.assertIn("Emit a complete action; do not search or read again", repair_contract)
        self.assertIn(
            "- none\nCurrently admissible completion schema",
            repair_contract,
        )

    def test_evidence_retriever_allows_open_super_bowl_event_anchor(
        self,
    ) -> None:
        question = "Who won Super Bowl XX?"
        evidence = (
            "Super Bowl XX matched the Chicago Bears and New England Patriots. "
            "The Bears defeated the Patriots by 46-10."
        )
        artifact = {
            "question_scope": question,
            "entity_identity": {
                "question_surface": "Super Bowl XX",
                "evidence_surface": "Super Bowl XX",
            },
            "target_relation": "defeated",
            "answer_type_constraint": "entity",
            "evidence_proposition": {
                "subject": "The Bears",
                "predicate": "defeated",
                "object_or_attribute_value": "the Patriots",
            },
            "evidence_span": evidence,
            "passage_id": "super-bowl-xx",
        }
        receipt = {
            "tool_id": QA_RETRIEVAL_TOOL_ID,
            "tool_version": "frozen-index-v1",
            "request": {
                "action": "read",
                "arguments": {"passage_id": "super-bowl-xx"},
            },
            "result": {
                "value": {
                    "operation": "read",
                    "passage_id": "super-bowl-xx",
                    "passage": {
                        "passage_id": "super-bowl-xx",
                        "title": "Super Bowl XX",
                        "text": evidence,
                    },
                },
                "completed": True,
            },
            "error_type": None,
        }

        issue = QARetrievalReactExecutionAdapter._evidence_retriever_completion_issue
        self.assertIsNone(
            issue(
                original_question=question,
                artifact=json.dumps(artifact),
                tool_receipts=[receipt],
            )
        )
        for answer_field in ("candidate_answer", "answer_slot", "final_answer"):
            self.assertNotIn(answer_field, artifact)

        both_arguments = json.loads(json.dumps(artifact))
        both_arguments["target_relation"] = "featured"
        both_arguments["evidence_proposition"] = {
            "subject": "Super Bowl XX",
            "predicate": "featured",
            "object_or_attribute_value": "Super Bowl XX",
        }
        both_arguments["evidence_span"] = "Super Bowl XX featured Super Bowl XX."
        both_receipt = json.loads(json.dumps(receipt))
        both_receipt["result"]["value"]["passage"]["text"] = both_arguments[
            "evidence_span"
        ]
        detail = issue(
            original_question=question,
            artifact=json.dumps(both_arguments),
            tool_receipts=[both_receipt],
        )
        self.assertIsNotNone(detail)
        assert detail is not None
        self.assertIn("must not bind", detail)
        self.assertIn("both relation arguments", detail)

    async def test_evidence_retriever_completes_one_grounded_read_artifact(
        self,
    ) -> None:
        question = "Which city does David Soul come from?"
        evidence = "David Soul is from Chicago."
        artifact = {
            "question_scope": question,
            "entity_identity": {
                "question_surface": "David Soul",
                "evidence_surface": "David Soul",
            },
            "target_relation": "is from",
            "answer_type_constraint": "entity",
            "evidence_proposition": {
                "subject": "David Soul",
                "predicate": "is from",
                "object_or_attribute_value": "Chicago",
            },
            "evidence_span": evidence,
            "passage_id": "p1",
        }

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

        class SoulIndex(FakeIndex):
            def read(self, passage_id: str) -> FakePassage:
                self.read_calls.append(passage_id)
                return FakePassage(passage_id, "d1", "David Soul", evidence)

        class SequenceGateway:
            def __init__(self) -> None:
                self.outputs = [
                    action("search", {"query": "David Soul origin city", "limit": 5}),
                    action("read", {"passage_id": "p1"}),
                    action("complete", {"value": artifact}),
                ]

            async def generate(self, request: AgentRequest) -> AgentResponse:
                del request
                return AgentResponse(self.outputs.pop(0))

        index = SoulIndex()
        response = await QARetrievalReactExecutionAdapter(
            gateway=SequenceGateway(),
            tool_registry=build_qa_tool_registry(index),
            max_turns=3,
            max_tool_calls=2,
            task_type="factual_qa",
            completion_policy="required_evidence",
        ).execute(
            AgentRequest(
                request_id="trivia:evidence-retriever-grounded",
                run_id="trivia",
                graph_revision=1,
                problem=question,
                agent=AgentNode(
                    "evidence_retriever",
                    "model",
                    "retrieve receipt-grounded entity and relation evidence",
                    role_family="evidence_retriever",
                    allowed_tools=(QA_RETRIEVAL_TOOL_ID,),
                    execution_mode="react",
                ),
                model=ModelSpec("model", "provider"),
                provider=ProviderSpec("provider", kind="test"),
                phase=ExecutionPhase.SINGLE,
                semantic_protocol=QA_VERIFIED_ANSWER_LINEAGE_PROTOCOL,
            )
        )

        self.assertEqual(
            json.dumps(
                artifact,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ),
            response.text,
        )
        self.assertEqual([("David Soul origin city", 5)], index.search_calls)
        self.assertEqual(["p1"], index.read_calls)
        self.assertEqual(2, response.metadata["tool_calls"])

    async def test_evidence_span_lexical_mismatch_repairs_on_preserved_read(
        self,
    ) -> None:
        question = "Who won Super Bowl XX?"
        evidence = (
            "Super Bowl XX matched the Chicago Bears and New England Patriots. "
            "The Bears defeated the Patriots by 46-10."
        )
        valid_artifact = {
            "question_scope": question,
            "entity_identity": {
                "question_surface": "Super Bowl XX",
                "evidence_surface": "Super Bowl XX",
            },
            "target_relation": "defeated",
            "answer_type_constraint": "entity",
            "evidence_proposition": {
                "subject": "The Bears",
                "predicate": "defeated",
                "object_or_attribute_value": "the Patriots",
            },
            "evidence_span": evidence,
            "passage_id": "p1",
        }
        invalid_artifact = {
            **valid_artifact,
            "evidence_span": (
                "Super Bowl XX was won when the Bears beat the Patriots 46-10."
            ),
        }

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

        class SuperBowlIndex(FakeIndex):
            def read(self, passage_id: str) -> FakePassage:
                self.read_calls.append(passage_id)
                return FakePassage(passage_id, "d1", "Super Bowl XX", evidence)

        class SequenceGateway:
            def __init__(self) -> None:
                self.requests: list[AgentRequest] = []
                self.outputs = [
                    action("search", {"query": "Super Bowl XX winner", "limit": 5}),
                    action("read", {"passage_id": "p1"}),
                    action("complete", {"value": invalid_artifact}),
                    action("complete", {"value": valid_artifact}),
                ]

            async def generate(self, request: AgentRequest) -> AgentResponse:
                self.requests.append(request)
                return AgentResponse(self.outputs.pop(0))

        index = SuperBowlIndex()
        gateway = SequenceGateway()
        response = await QARetrievalReactExecutionAdapter(
            gateway=gateway,
            tool_registry=build_qa_tool_registry(index),
            max_turns=4,
            max_tool_calls=10,
            task_type="factual_qa",
            completion_policy="required_evidence",
        ).execute(
            AgentRequest(
                request_id="trivia:super-bowl-span-repair",
                run_id="trivia",
                graph_revision=1,
                problem=question,
                agent=AgentNode(
                    "evidence_retriever",
                    "model",
                    "retrieve receipt-grounded entity and relation evidence",
                    role_family="evidence_retriever",
                    allowed_tools=(QA_RETRIEVAL_TOOL_ID,),
                    execution_mode="react",
                ),
                model=ModelSpec("model", "provider"),
                provider=ProviderSpec("provider", kind="test"),
                phase=ExecutionPhase.SINGLE,
                semantic_protocol=QA_VERIFIED_ANSWER_LINEAGE_PROTOCOL,
            )
        )

        self.assertEqual([("Super Bowl XX winner", 5)], index.search_calls)
        self.assertEqual(["p1"], index.read_calls)
        self.assertEqual(2, response.metadata["tool_calls"])
        lexical_feedback = response.metadata["react_trace"][2][
            "public_error_code"
        ]
        self.assertTrue(lexical_feedback.startswith("qa_semantic_artifact_invalid:"))
        self.assertIn("typography-canonical lexical match", lexical_feedback)
        self.assertNotIn("knowledge_base_coverage_failure", lexical_feedback)
        repair_contract = gateway.requests[3].agent.contract
        self.assertIn("Preserve the cited passage_id", repair_contract)
        self.assertIn("do not paraphrase", repair_contract)
        self.assertIn(
            "- none\nCurrently admissible completion schema",
            repair_contract,
        )

    async def test_evidence_retriever_repairs_relation_surface_on_preserved_read(
        self,
    ) -> None:
        question = "Where in England was Dame Judi Dench born?"
        evidence = "Dench was born in Heworth, North Riding of Yorkshire."
        base_artifact = {
            "question_scope": question,
            "entity_identity": {
                "question_surface": "Dench",
                "evidence_surface": "Dench",
            },
            "answer_type_constraint": "location",
            "evidence_proposition": {
                "subject": "Dench",
                "predicate": "was born in",
                "object_or_attribute_value": "Heworth, North Riding of Yorkshire",
            },
            "evidence_span": evidence,
            "passage_id": "p1",
        }

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

        class DenchIndex(FakeIndex):
            def read(self, passage_id: str) -> FakePassage:
                self.read_calls.append(passage_id)
                return FakePassage(passage_id, "d1", "Judi Dench", evidence)

        class SequenceGateway:
            def __init__(self) -> None:
                self.requests: list[AgentRequest] = []
                self.outputs = [
                    action(
                        "search",
                        {"query": "Dame Judi Dench birthplace", "limit": 5},
                    ),
                    action("read", {"passage_id": "p1"}),
                    action(
                        "complete",
                        {
                            "value": {
                                **base_artifact,
                                "entity_identity": {
                                    "question_surface": question,
                                    "evidence_surface": evidence,
                                },
                                "target_relation": "birthplace",
                            }
                        },
                    ),
                    action(
                        "complete",
                        {
                            "value": {
                                **base_artifact,
                                "entity_identity": {
                                    "question_surface": "Dench",
                                    "evidence_surface": evidence,
                                },
                                "target_relation": "birthplace",
                            }
                        },
                    ),
                    action(
                        "complete",
                        {"value": {**base_artifact, "target_relation": "birthplace"}},
                    ),
                    action(
                        "complete",
                        {"value": {**base_artifact, "target_relation": "was born in"}},
                    ),
                ]

            async def generate(self, request: AgentRequest) -> AgentResponse:
                self.requests.append(request)
                return AgentResponse(self.outputs.pop(0))

        index = DenchIndex()
        gateway = SequenceGateway()
        response = await QARetrievalReactExecutionAdapter(
            gateway=gateway,
            tool_registry=build_qa_tool_registry(index),
            max_turns=6,
            max_tool_calls=10,
            task_type="factual_qa",
            completion_policy="required_evidence",
        ).execute(
            AgentRequest(
                request_id="trivia:evidence-retriever-relation-repair",
                run_id="trivia",
                graph_revision=1,
                problem=question,
                agent=AgentNode(
                    "evidence_retriever",
                    "model",
                    "retrieve receipt-grounded entity and relation evidence",
                    role_family="evidence_retriever",
                    allowed_tools=(QA_RETRIEVAL_TOOL_ID,),
                    execution_mode="react",
                ),
                model=ModelSpec("model", "provider"),
                provider=ProviderSpec("provider", kind="test"),
                phase=ExecutionPhase.SINGLE,
                semantic_protocol=QA_VERIFIED_ANSWER_LINEAGE_PROTOCOL,
            )
        )

        self.assertEqual([("Dame Judi Dench birthplace", 5)], index.search_calls)
        self.assertEqual(["p1"], index.read_calls)
        public_errors = [
            response.metadata["react_trace"][index]["public_error_code"]
            for index in (2, 3, 4)
        ]
        self.assertTrue(
            all(
                error.startswith("qa_semantic_artifact_invalid:")
                for error in public_errors
            )
        )
        self.assertIn("whole original question", public_errors[0])
        self.assertIn("whole evidence_span", public_errors[1])
        self.assertIn("target_relation does not occur", public_errors[2])
        for request in gateway.requests[3:6]:
            self.assertIn("do not add a search or read", request.agent.contract)
        self.assertEqual(2, response.metadata["tool_calls"])

    def test_evidence_retriever_rejects_unprovenanced_or_answer_only_artifacts(
        self,
    ) -> None:
        question = "Which city does David Soul come from?"
        evidence = "David Soul comes from Chicago."
        receipt = {
            "tool_id": QA_RETRIEVAL_TOOL_ID,
            "tool_version": "frozen-index-v1",
            "request": {
                "action": "read",
                "arguments": {"passage_id": "p1"},
            },
            "result": {
                "value": {
                    "operation": "read",
                    "passage_id": "p1",
                    "passage": {"passage_id": "p1", "text": evidence},
                },
                "completed": True,
            },
            "error_type": None,
        }
        valid = {
            "question_scope": question,
            "entity_identity": {
                "question_surface": "David Soul",
                "evidence_surface": "David Soul",
            },
            "target_relation": "comes from",
            "answer_type_constraint": "entity",
            "evidence_proposition": {
                "subject": "David Soul",
                "predicate": "comes from",
                "object_or_attribute_value": "Chicago",
            },
            "evidence_span": evidence,
            "passage_id": "p1",
        }

        issue = QARetrievalReactExecutionAdapter._evidence_retriever_completion_issue
        self.assertIsNone(
            issue(
                original_question=question,
                artifact=json.dumps(valid),
                tool_receipts=[receipt],
            )
        )

        invalid_cases: dict[str, tuple[object, str]] = {}
        for name, path, value, expected in (
            (
                "passage",
                ("passage_id",),
                "p2",
                "no matching successful",
            ),
            (
                "span",
                ("evidence_span",),
                "David Soul comes from London.",
                "no typography-canonical lexical match",
            ),
            (
                "question entity",
                ("entity_identity", "question_surface"),
                "David Crockett",
                "does not occur in the original question",
            ),
            (
                "evidence entity",
                ("entity_identity", "evidence_surface"),
                "David Crockett",
                "does not occur in evidence_span",
            ),
            (
                "relation",
                ("target_relation",),
                "was born in",
                "target_relation does not occur",
            ),
        ):
            candidate = json.loads(json.dumps(valid))
            target = candidate
            for key in path[:-1]:
                target = target[key]
            target[path[-1]] = value
            invalid_cases[name] = (candidate, expected)
        invalid_cases["bare answer"] = (
            "London",
            "labelled artifact must begin with a declared `Field:` label",
        )

        for name, (candidate, expected) in invalid_cases.items():
            with self.subTest(name=name):
                detail = issue(
                    original_question=question,
                    artifact=(
                        candidate
                        if isinstance(candidate, str)
                        else json.dumps(candidate)
                    ),
                    tool_receipts=[receipt],
                )
                self.assertIsNotNone(detail)
                assert detail is not None
                self.assertIn(expected, detail)

    def test_evidence_retriever_alias_requires_explicit_identity_binding(
        self,
    ) -> None:
        question = "Where was Norma Jeane Mortenson born?"

        def receipt(passage_id: str, text: str) -> dict[str, object]:
            return {
                "tool_id": QA_RETRIEVAL_TOOL_ID,
                "tool_version": "frozen-index-v1",
                "request": {
                    "action": "read",
                    "arguments": {"passage_id": passage_id},
                },
                "result": {
                    "value": {
                        "operation": "read",
                        "passage_id": passage_id,
                        "passage": {
                            "passage_id": passage_id,
                            "text": text,
                        },
                    },
                    "completed": True,
                },
                "error_type": None,
            }

        positive_text = (
            "Norma Jeane Mortenson, known as Marilyn Monroe, was born in "
            "Los Angeles."
        )
        positive = {
            "question_scope": question,
            "entity_identity": {
                "question_surface": "Norma Jeane Mortenson",
                "evidence_surface": "Marilyn Monroe",
            },
            "target_relation": "was born in",
            "answer_type_constraint": "location",
            "evidence_proposition": {
                "subject": "Marilyn Monroe",
                "predicate": "was born in",
                "object_or_attribute_value": "Los Angeles",
            },
            "evidence_span": positive_text,
            "passage_id": "positive",
        }
        issue = QARetrievalReactExecutionAdapter._evidence_retriever_completion_issue
        self.assertIsNone(
            issue(
                original_question=question,
                artifact=json.dumps(positive),
                tool_receipts=[receipt("positive", positive_text)],
            )
        )

        negative_text = "Marilyn Monroe was born in Los Angeles."
        negative = {
            **positive,
            "evidence_span": negative_text,
            "passage_id": "negative",
        }
        detail = issue(
            original_question=question,
            artifact=json.dumps(negative),
            tool_receipts=[receipt("negative", negative_text)],
        )
        self.assertIsNotNone(detail)
        assert detail is not None
        self.assertIn("alias identity lacks an explicit binding", detail)

    def test_evidence_retriever_binds_relation_argument_to_question_answer_type(
        self,
    ) -> None:
        def receipt(passage_id: str, text: str) -> dict[str, object]:
            return {
                "tool_id": QA_RETRIEVAL_TOOL_ID,
                "tool_version": "frozen-index-v1",
                "request": {
                    "action": "read",
                    "arguments": {"passage_id": passage_id},
                },
                "result": {
                    "value": {
                        "operation": "read",
                        "passage_id": passage_id,
                        "passage": {"passage_id": passage_id, "text": text},
                    },
                    "completed": True,
                },
                "error_type": None,
            }

        issue = QARetrievalReactExecutionAdapter._evidence_retriever_completion_issue
        question = "Where in England was Dame Judi Dench born?"
        date_evidence = "Dench was born 9 December 1934."
        date_artifact = {
            "question_scope": question,
            "entity_identity": {
                "question_surface": "Dench",
                "evidence_surface": "Dench",
            },
            "target_relation": "was born",
            "answer_type_constraint": "location",
            "evidence_proposition": {
                "subject": "Dench",
                "predicate": "was born",
                "object_or_attribute_value": "9 December 1934",
            },
            "evidence_span": date_evidence,
            "passage_id": "date",
        }
        detail = issue(
            original_question=question,
            artifact=json.dumps(date_artifact),
            tool_receipts=[receipt("date", date_evidence)],
        )
        self.assertIsNotNone(detail)
        assert detail is not None
        self.assertIn("supplies a date relation argument", detail)
        self.assertIn("requested location relation argument", detail)

        predicate_as_entity = json.loads(json.dumps(date_artifact))
        predicate_as_entity["entity_identity"] = {
            "question_surface": "born",
            "evidence_surface": "born",
        }
        predicate_as_entity["target_relation"] = "born"
        predicate_as_entity["evidence_proposition"]["predicate"] = "born"
        detail = issue(
            original_question=question,
            artifact=json.dumps(predicate_as_entity),
            tool_receipts=[receipt("date", date_evidence)],
        )
        self.assertIsNotNone(detail)
        assert detail is not None
        self.assertIn("entity surface must not be the relation predicate", detail)

        place_evidence = (
            "Dench was born in Heworth, North Riding of Yorkshire."
        )
        place_artifact = {
            **date_artifact,
            "target_relation": "was born in",
            "evidence_proposition": {
                "subject": "Dench",
                "predicate": "was born in",
                "object_or_attribute_value": (
                    "Heworth, North Riding of Yorkshire"
                ),
            },
            "evidence_span": place_evidence,
            "passage_id": "place",
        }
        self.assertIsNone(
            issue(
                original_question=question,
                artifact=json.dumps(place_artifact),
                tool_receipts=[receipt("place", place_evidence)],
            )
        )

    def test_evidence_retriever_city_and_date_constraints_are_question_only(
        self,
    ) -> None:
        def validate(
            *, question: str, evidence: str, answer_type: str, value: str
        ) -> str | None:
            passage_id = answer_type
            artifact = {
                "question_scope": question,
                "entity_identity": {
                    "question_surface": "conference",
                    "evidence_surface": "conference",
                },
                "target_relation": "was held in" if answer_type == "location" else "was held on",
                "answer_type_constraint": answer_type,
                "evidence_proposition": {
                    "subject": "conference",
                    "predicate": (
                        "was held in" if answer_type == "location" else "was held on"
                    ),
                    "object_or_attribute_value": value,
                },
                "evidence_span": evidence,
                "passage_id": passage_id,
            }
            receipt = {
                "tool_id": QA_RETRIEVAL_TOOL_ID,
                "tool_version": "frozen-index-v1",
                "request": {
                    "action": "read",
                    "arguments": {"passage_id": passage_id},
                },
                "result": {
                    "value": {
                        "operation": "read",
                        "passage_id": passage_id,
                        "passage": {
                            "passage_id": passage_id,
                            "text": evidence,
                        },
                    },
                    "completed": True,
                },
                "error_type": None,
            }
            return QARetrievalReactExecutionAdapter._evidence_retriever_completion_issue(
                original_question=question,
                artifact=json.dumps(artifact),
                tool_receipts=[receipt],
            )

        self.assertIsNone(
            validate(
                question="What city was the conference held in?",
                evidence="The conference was held in Kyoto.",
                answer_type="location",
                value="Kyoto",
            )
        )
        self.assertIsNone(
            validate(
                question="When was the conference held?",
                evidence="The conference was held on 12 March 2020.",
                answer_type="date",
                value="12 March 2020",
            )
        )
        mismatch = validate(
            question="When was the conference held?",
            evidence="The conference was held on Kyoto.",
            answer_type="date",
            value="Kyoto",
        )
        self.assertIsNotNone(mismatch)
        assert mismatch is not None
        self.assertIn("does not supply the question's requested date", mismatch)

    async def test_wrong_relation_argument_type_advances_retrieval_strategy(
        self,
    ) -> None:
        question = "Where in England was Dame Judi Dench born?"
        evidence = "Dench was born 9 December 1934."
        invalid_artifact = {
            "question_scope": question,
            "entity_identity": {
                "question_surface": "Dench",
                "evidence_surface": "Dench",
            },
            "target_relation": "was born",
            "answer_type_constraint": "location",
            "evidence_proposition": {
                "subject": "Dench",
                "predicate": "was born",
                "object_or_attribute_value": "9 December 1934",
            },
            "evidence_span": evidence,
            "passage_id": "p1",
        }

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

        class DenchDateIndex(FakeIndex):
            def search(self, query: str, *, limit: int) -> tuple[FakeHit, ...]:
                self.search_calls.append((query, limit))
                passage_id = "p1" if len(self.search_calls) == 1 else "p2"
                return (FakeHit(passage_id, passage_id, "Judi Dench", evidence, 1),)

            def read(self, passage_id: str) -> FakePassage:
                self.read_calls.append(passage_id)
                return FakePassage(passage_id, passage_id, "Judi Dench", evidence)

        class SequenceGateway:
            def __init__(self) -> None:
                self.requests: list[AgentRequest] = []
                self.outputs = [
                    action("search", {"query": "Judi Dench born", "limit": 5}),
                    action("read", {"passage_id": "p1"}),
                    action("complete", {"value": invalid_artifact}),
                    action(
                        "search",
                        {"query": "Judi Dench birthplace England", "limit": 10},
                    ),
                ]

            async def generate(self, request: AgentRequest) -> AgentResponse:
                self.requests.append(request)
                return AgentResponse(self.outputs.pop(0))

        index = DenchDateIndex()
        gateway = SequenceGateway()
        with self.assertRaises(ReactExecutionError) as caught:
            await QARetrievalReactExecutionAdapter(
                gateway=gateway,
                tool_registry=build_qa_tool_registry(index),
                max_turns=4,
                max_tool_calls=10,
                task_type="factual_qa",
                completion_policy="required_evidence",
            ).execute(
                AgentRequest(
                    request_id="trivia:dench-answer-type-recovery",
                    run_id="trivia",
                    graph_revision=1,
                    problem=question,
                    agent=AgentNode(
                        "evidence_retriever",
                        "model",
                        "retrieve the requested relation argument",
                        role_family="evidence_retriever",
                        allowed_tools=(QA_RETRIEVAL_TOOL_ID,),
                        execution_mode="react",
                    ),
                    model=ModelSpec("model", "provider"),
                    provider=ProviderSpec("provider", kind="test"),
                    phase=ExecutionPhase.SINGLE,
                    semantic_protocol=QA_VERIFIED_ANSWER_LINEAGE_PROTOCOL,
                )
            )

        self.assertEqual(
            [
                ("Judi Dench born", 5),
                ("Judi Dench birthplace England", 10),
            ],
            index.search_calls,
        )
        feedback = caught.exception.react_trace[2]["public_error_code"]
        self.assertTrue(
            feedback.startswith("qa_semantic_evidence_provenance_invalid:")
        )
        self.assertIn("supplies a date relation argument", feedback)
        self.assertIn("`spelling_normalization`", gateway.requests[3].agent.contract)

    async def test_irrelevant_david_crockett_read_advances_david_soul_retrieval(
        self,
    ) -> None:
        question = "Which city does David Soul come from?"
        crockett_evidence = (
            "David Crockett was born in Greene County, Tennessee."
        )
        invalid_artifact = {
            "question_scope": question,
            "entity_identity": {
                "question_surface": "David Soul",
                "evidence_surface": "David Crockett",
            },
            "target_relation": "was born in",
            "answer_type_constraint": "entity",
            "evidence_proposition": {
                "subject": "David Crockett",
                "predicate": "was born in",
                "object_or_attribute_value": "Greene County, Tennessee",
            },
            "evidence_span": crockett_evidence,
            "passage_id": "p1",
        }

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

        class DisambiguationIndex(FakeIndex):
            def search(self, query: str, *, limit: int) -> tuple[FakeHit, ...]:
                self.search_calls.append((query, limit))
                passage_id = "p1" if len(self.search_calls) == 1 else "p2"
                title = "David Crockett" if passage_id == "p1" else "David Soul"
                return (FakeHit(passage_id, passage_id, title, title, 1),)

            def read(self, passage_id: str) -> FakePassage:
                self.read_calls.append(passage_id)
                return FakePassage(
                    passage_id,
                    passage_id,
                    "David Crockett",
                    crockett_evidence,
                )

        class SequenceGateway:
            def __init__(self) -> None:
                self.outputs = [
                    action("search", {"query": "David Soul birthplace", "limit": 5}),
                    action("read", {"passage_id": "p1"}),
                    action("complete", {"value": invalid_artifact}),
                    action(
                        "search",
                        {"query": "David Soul origin city", "limit": 10},
                    ),
                ]
                self.requests: list[AgentRequest] = []

            async def generate(self, request: AgentRequest) -> AgentResponse:
                self.requests.append(request)
                return AgentResponse(self.outputs.pop(0))

        index = DisambiguationIndex()
        gateway = SequenceGateway()
        request = AgentRequest(
            request_id="trivia:david-soul-retrieval-recovery",
            run_id="trivia",
            graph_revision=1,
            problem=question,
            agent=AgentNode(
                "evidence_retriever",
                "model",
                "retrieve evidence for the original entity and relation",
                role_family="evidence_retriever",
                allowed_tools=(QA_RETRIEVAL_TOOL_ID,),
                execution_mode="react",
            ),
            model=ModelSpec("model", "provider"),
            provider=ProviderSpec("provider", kind="test"),
            phase=ExecutionPhase.SINGLE,
            semantic_protocol=QA_VERIFIED_ANSWER_LINEAGE_PROTOCOL,
        )

        with self.assertRaises(ReactExecutionError) as caught:
            await QARetrievalReactExecutionAdapter(
                gateway=gateway,
                tool_registry=build_qa_tool_registry(index),
                max_turns=4,
                max_tool_calls=10,
                task_type="factual_qa",
                completion_policy="required_evidence",
            ).execute(request)

        self.assertEqual(
            [
                ("David Soul birthplace", 5),
                ("David Soul origin city", 10),
            ],
            index.search_calls,
        )
        invalid_feedback = caught.exception.react_trace[2]["public_error_code"]
        self.assertTrue(
            invalid_feedback.startswith(
                "qa_semantic_evidence_provenance_invalid:"
            )
        )
        self.assertIn("alias identity lacks an explicit binding", invalid_feedback)
        retry_schema = json.loads(
            gateway.requests[3].model.metadata["response_json_schema"]
        )
        self.assertEqual(
            10,
            retry_schema["properties"]["arguments"]["properties"]["limit"][
                "const"
            ],
        )
        self.assertIn("`spelling_normalization`", gateway.requests[3].agent.contract)

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
