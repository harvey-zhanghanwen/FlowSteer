from __future__ import annotations

from dataclasses import FrozenInstanceError
import json
import unittest

from src.interactive.agent_graph import AgentGraph, AgentNode, AgentRelation
from src.interactive.agent_runtime import (
    AgentRequest,
    AgentResponse,
    AgentRuntime,
    ReasoningExecutionAdapter,
)
from src.interactive.agent_workflow_env import (
    AgentWorkflowEnv,
    AgentWorkflowStateError,
    _HOTPOTQA_FORMAT_CONTRACT,
)
from src.interactive.model_registry import ModelRegistry, ModelSpec, ProviderSpec
from src.interactive.qa_tool_adapter import build_qa_tool_registry


def _registry() -> ModelRegistry:
    provider = ProviderSpec("fake", kind="test", api_key_env="FAKE_API_KEY")
    return ModelRegistry(
        [provider],
        [
            ModelSpec("reader-model", "fake"),
            ModelSpec("reasoner-model", "fake"),
            ModelSpec("verifier-model", "fake"),
            ModelSpec("formatter-model", "fake"),
        ],
    )


def _semantic_graph() -> AgentGraph:
    return AgentGraph(
        [
            AgentNode(
                "reader",
                "reader-model",
                "retrieve public evidence",
                role_family="evidence_retriever",
                allowed_tools=("qa-retrieval",),
                execution_mode="react",
                artifact_type="retrieval_evidence",
            ),
            AgentNode(
                "reasoner",
                "reasoner-model",
                "bind retrieved evidence to the question answer slot",
                role_family="reasoner",
                allowed_tools=(),
                execution_mode="reasoning",
                artifact_type="semantic_candidate",
            ),
            AgentNode(
                "verifier",
                "verifier-model",
                "verify evidence, entity binding, relation, and question scope",
                role_family="verifier",
                artifact_type="verified_semantic_answer",
            ),
            AgentNode(
                "formatter",
                "formatter-model",
                _HOTPOTQA_FORMAT_CONTRACT,
                role_family="format",
                artifact_type="answer_wrapper",
            ),
        ],
        [
            AgentRelation("reader", "reasoner", True, False),
            AgentRelation("reasoner", "verifier", True, False),
            AgentRelation("verifier", "formatter", True, False),
        ],
        output_agent_id="formatter",
    )


class _NoopRetrievalIndex:
    manifest = type(
        "Manifest",
        (),
        {
            "corpus_name": "test-public-corpus",
            "corpus_version": "test-v1",
            "index_id": "test-index-v1",
            "format": "skillev-public-retrieval-index@2",
            "retrieval_backend": "test",
        },
    )()

    def search(self, query: str, *, limit: int) -> tuple[object, ...]:
        del query, limit
        return ()

    def read(self, passage_id: str) -> object:
        raise AssertionError(f"unexpected test retrieval read {passage_id!r}")


class _TriviaSemanticGateway:
    def __init__(
        self,
        *,
        coverage_failure: bool = False,
        ungrounded_subject: bool = False,
    ) -> None:
        self.coverage_failure = coverage_failure
        self.ungrounded_subject = ungrounded_subject

    async def generate(self, request: AgentRequest):  # type: ignore[no-untyped-def]
        if request.agent.id == "reader":
            return AgentResponse(
                json.dumps(
                    {
                        "question_scope": request.problem,
                        "entity_identity": {
                            "question_surface": "France",
                            "evidence_surface": "France",
                        },
                        "target_relation": "capital",
                        "answer_type_constraint": "short_answer",
                        "evidence_proposition": {
                            "subject": "France",
                            "predicate": "capital",
                            "object_or_attribute_value": "Paris",
                        },
                        "evidence_span": "Paris is the capital of France.",
                        "passage_id": "p1",
                    }
                ),
                {
                    "tool_receipts": [
                        {
                            "tool_id": "qa-retrieval",
                            "tool_version": "test-v1",
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
                                        "text": "Paris is the capital of France.",
                                    },
                                },
                                "completed": True,
                            },
                            "error_type": None,
                        }
                    ]
                },
            )
        if request.agent.id == "reasoner":
            if self.coverage_failure:
                return json.dumps(
                    {
                        "failure_type": "knowledge_base_coverage_failure",
                        "operational_diagnosis": "bounded retrieval returned no admissible evidence",
                    }
                )
            return json.dumps(
                {
                    "question_scope": request.problem,
                    "answer_slot": {
                        "answer_type": "short_answer",
                        "answer_cardinality": "single",
                        "qualifiers": [],
                        "proposition_index": 0,
                        "answer_field": "object_or_attribute_value",
                    },
                    "evidence_propositions": [
                        {
                            "subject": (
                                "Germany" if self.ungrounded_subject else "France"
                            ),
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
            )
        if request.agent.id == "verifier":
            return json.dumps(
                {
                    "candidate_answer": "Paris",
                    "evidence_supported": True,
                    "entity_attribute_binding_correct": True,
                    "alias_binding_correct": True,
                    "answer_type_cardinality_correct": True,
                    "multi_hop_complete": True,
                    "minimal_answer_surface": True,
                    "scope_preserved": True,
                    "verification_status": "supported",
                }
            )
        if request.agent.id == "formatter":
            return "<answer>Paris</answer>"
        if request.agent.id == "output":
            return AgentResponse(
                "<answer>Paris</answer>",
                {
                    "tool_receipts": [
                        {
                            "tool_id": "qa-retrieval",
                            "tool_version": "test-v1",
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
                                            "Paris is the capital of France."
                                        ),
                                    },
                                },
                                "completed": True,
                            },
                            "error_type": None,
                        }
                    ]
                },
            )
        raise AssertionError(f"unexpected Agent {request.agent.id!r}")


def _runtime(
    registry: ModelRegistry,
    gateway: _TriviaSemanticGateway,
    *,
    dataset_id: str,
) -> AgentRuntime:
    return AgentRuntime(
        registry,
        gateway,
        execution_adapters={"react": ReasoningExecutionAdapter(gateway)},
        tool_registry=build_qa_tool_registry(_NoopRetrievalIndex()),
        dataset_id=dataset_id,
        semantic_protocol="qa_verified_answer_lineage_v2",
    )


def _env(
    *,
    dataset_id: str = "triviaqa",
    coverage_failure: bool = False,
    ungrounded_subject: bool = False,
) -> AgentWorkflowEnv:
    registry = _registry()
    gateway = _TriviaSemanticGateway(
        coverage_failure=coverage_failure,
        ungrounded_subject=ungrounded_subject,
    )
    return AgentWorkflowEnv(
        registry,
        runtime=_runtime(registry, gateway, dataset_id=dataset_id),
        graph=_semantic_graph(),
        problem="What is the capital of France?",
        execute_on_edit=True,
        require_exact_answer_tag=True,
        require_format_agent=True,
        semantic_protocol="qa_verified_answer_lineage_v2",
        recovery_policy="preserve_diagnose_repair_augment",
        required_evidence_tool_id="qa-retrieval",
    )


class SharedQASemanticLineageTests(unittest.IsolatedAsyncioTestCase):
    async def test_role_conditional_generic_output_finishes_without_named_spine(
        self,
    ) -> None:
        registry = _registry()
        gateway = _TriviaSemanticGateway()
        graph = AgentGraph(
            [
                AgentNode(
                    "output",
                    "reader-model",
                    "answer from successful public retrieval evidence",
                    role_family="output",
                    allowed_tools=("qa-retrieval",),
                    execution_mode="react",
                    artifact_type="answer_wrapper",
                )
            ],
            output_agent_id="output",
        )
        env = AgentWorkflowEnv(
            registry,
            runtime=_runtime(registry, gateway, dataset_id="triviaqa"),
            graph=graph,
            problem="What is the capital of France?",
            execute_on_edit=True,
            require_exact_answer_tag=True,
            require_format_agent=False,
            semantic_protocol="qa_verified_answer_lineage_v2",
            recovery_policy="preserve_diagnose_repair_augment",
            required_evidence_tool_id="qa-retrieval",
        )

        executed = await env.step(
            '{"action":"modify_agent","agent_id":"output",'
            '"contract":"return the short answer grounded in a successful read"}'
        )

        self.assertTrue(executed.accepted)
        self.assertEqual((), env._missing_semantic_role_families())
        self.assertEqual((), env._required_semantic_edges())
        self.assertTrue(env.finish_admissibility()["admissible"])
        finished = await env.step('{"action":"finish"}')
        self.assertTrue(finished.done)
        self.assertEqual("<answer>Paris</answer>", finished.final_answer)

    async def test_trivia_accepts_one_proposition_and_captures_atomic_lineage(
        self,
    ) -> None:
        env = _env(dataset_id="triviaqa")

        executed = await env.step(
            '{"action":"modify_agent","agent_id":"reader",'
            '"contract":"retrieve and preserve public evidence"}'
        )

        self.assertTrue(executed.accepted)
        self.assertTrue(env.finish_admissibility()["admissible"])
        self.assertEqual(("finish",), env.model_admissible_action_types())
        lineage = env.last_valid_evidence_lineage
        self.assertIsNotNone(lineage)
        assert lineage is not None
        self.assertEqual("<answer>Paris</answer>", lineage.answer)
        self.assertIs(executed.execution, lineage.runtime)
        self.assertEqual(env.revision, lineage.graph_revision)
        self.assertEqual(env.graph.snapshot(), lineage.graph_snapshot)
        with self.assertRaises(FrozenInstanceError):
            lineage.graph_revision = 999  # type: ignore[misc]

        rejected = await env.step(
            '{"action":"modify_agent","agent_id":"verifier",'
            '"contract":"replace the already verified answer"}'
        )
        self.assertFalse(rejected.accepted)
        self.assertIs(lineage, env.last_valid_evidence_lineage)

    async def test_hotpot_v2_keeps_two_proposition_minimum(self) -> None:
        env = _env(dataset_id="hotpotqa")

        executed = await env.step(
            '{"action":"modify_agent","agent_id":"reader",'
            '"contract":"retrieve and preserve public evidence"}'
        )

        self.assertTrue(executed.accepted)
        admission = env.finish_admissibility()
        self.assertFalse(admission["admissible"])
        self.assertIn("at least two propositions", admission["reason"])
        self.assertIsNone(env.last_valid_evidence_lineage)

    async def test_coverage_failure_has_retrieval_database_attribution(self) -> None:
        env = _env(dataset_id="triviaqa", coverage_failure=True)

        executed = await env.step(
            '{"action":"modify_agent","agent_id":"reader",'
            '"contract":"retrieve and preserve public evidence"}'
        )

        self.assertTrue(executed.accepted)
        admission = env.finish_admissibility()
        self.assertFalse(admission["admissible"])
        self.assertEqual("semantic_protocol", admission["stage"])
        self.assertIn("knowledge_base_coverage_failure", admission["reason"])
        self.assertEqual(
            "retrieval_or_database_coverage",
            admission["failure_attribution"]["responsible_constraint"],
        )
        self.assertFalse(
            admission["failure_attribution"]["corpus_level_oracle_claim"]
        )
        self.assertIsNone(env.last_valid_evidence_lineage)

    async def test_v2_rejects_ungrounded_answer_bearing_entity_binding(self) -> None:
        env = _env(dataset_id="triviaqa", ungrounded_subject=True)

        executed = await env.step(
            '{"action":"modify_agent","agent_id":"reader",'
            '"contract":"retrieve and preserve public evidence"}'
        )

        self.assertTrue(executed.accepted)
        admission = env.finish_admissibility()
        self.assertFalse(admission["admissible"])
        self.assertIn("subject is not grounded", admission["reason"])
        self.assertIn("evidence_propositions[0].subject", admission["reason"])
        self.assertIsNone(env.last_valid_evidence_lineage)

    def test_v2_requires_an_explicit_supported_dataset_id(self) -> None:
        registry = _registry()
        gateway = _TriviaSemanticGateway()
        runtime = _runtime(registry, gateway, dataset_id="webshop")

        with self.assertRaisesRegex(
            AgentWorkflowStateError,
            "runtime.dataset_id",
        ):
            AgentWorkflowEnv(
                registry,
                runtime=runtime,
                problem="What is the capital of France?",
                semantic_protocol="qa_verified_answer_lineage_v2",
                recovery_policy="preserve_diagnose_repair_augment",
                required_evidence_tool_id="qa-retrieval",
            )


if __name__ == "__main__":
    unittest.main()
