from __future__ import annotations

import json
from types import SimpleNamespace
import unittest

from src.interactive.agent_action_parser import AgentAction, AgentActionType
from src.interactive.agent_graph import AgentGraph, AgentNode, AgentRelation
from src.interactive.agent_runtime import (
    AgentRequest,
    AgentRuntime,
    AgentRuntimeResult,
    ExecutionPhase,
    UpstreamMessage,
)
from src.interactive.agent_workflow_env import (
    AgentWorkflowEnv,
    _HOTPOTQA_FORMAT_CONTRACT,
    _QA_MEMORY_EVIDENCE_RETRIEVER_RECOVERY_COMPLETION,
)
from src.interactive.model_registry import ModelRegistry, ModelSpec, ProviderSpec
from src.interactive.openai_gateway import build_agent_messages
from src.interactive.qa_tool_adapter import (
    QA_VERIFIED_ANSWER_LINEAGE_PROTOCOL,
    QARetrievalReactExecutionAdapter,
    TRIVIAQA_QA_MEMORY_TOOL_ID,
    _question_entity_anchor_tokens,
    _surface_binds_entity_anchor,
    build_qa_tool_registry,
)


_QUESTION = "Which author wrote the novel?"
_ROWS = (
    ("memory-001", "triviaqa:tc_129", "Ada", 0.93),
    ("memory-002", "triviaqa:tc_130", "Grace", 0.88),
    ("memory-003", "triviaqa:tc_131", "Charles", 0.81),
)


class _UnusedGateway:
    async def generate(self, request: object) -> str:
        raise AssertionError(f"unexpected model request: {request!r}")


def _registry() -> ModelRegistry:
    return ModelRegistry(
        [ProviderSpec("fake", kind="test")],
        [ModelSpec("fake-model", "fake")],
    )


def _graph() -> AgentGraph:
    return AgentGraph(
        [
            AgentNode(
                "retriever",
                "fake-model",
                (
                    "ground evidence for the original entity and requested "
                    "relation in matching successful Tool receipts"
                ),
                role_family="evidence_retriever",
                execution_mode="react",
                allowed_tools=(TRIVIAQA_QA_MEMORY_TOOL_ID,),
                artifact_type="retrieval_evidence",
            ),
            AgentNode(
                "reasoner",
                "fake-model",
                (
                    "bind grounded evidence to the original answer slot and "
                    "requested relation, then derive one semantic candidate"
                ),
                role_family="reasoner",
                execution_mode="reasoning",
                allowed_tools=(),
                artifact_type="semantic_candidate",
            ),
            AgentNode(
                "verifier",
                "fake-model",
                (
                    "check entity identity, Tool provenance, semantic scope, "
                    "relation binding, and answer lineage without changing "
                    "the candidate"
                ),
                role_family="verifier",
                execution_mode="reasoning",
                allowed_tools=(),
                artifact_type="verified_semantic_answer",
            ),
            AgentNode(
                "formatter",
                "fake-model",
                _HOTPOTQA_FORMAT_CONTRACT,
                role_family="format",
                execution_mode="reasoning",
                allowed_tools=(),
                artifact_type="answer_wrapper",
            ),
        ],
        [
            AgentRelation("retriever", "reasoner", True, False),
            AgentRelation("reasoner", "verifier", True, False),
            AgentRelation("verifier", "formatter", True, False),
        ],
        output_agent_id="formatter",
    )


def _env(
    *,
    parametric_fallback_after_coverage_failure: bool = False,
) -> AgentWorkflowEnv:
    registry = _registry()
    runtime = AgentRuntime(
        registry,
        _UnusedGateway(),
        dataset_id="triviaqa",
        semantic_protocol=QA_VERIFIED_ANSWER_LINEAGE_PROTOCOL,
    )
    return AgentWorkflowEnv(
        registry,
        runtime=runtime,
        graph=_graph(),
        problem=_QUESTION,
        execute_on_edit=False,
        require_exact_answer_tag=True,
        require_format_agent=True,
        semantic_protocol=QA_VERIFIED_ANSWER_LINEAGE_PROTOCOL,
        recovery_policy="preserve_diagnose_repair_augment",
        required_evidence_tool_id=TRIVIAQA_QA_MEMORY_TOOL_ID,
        parametric_fallback_after_coverage_failure=(
            parametric_fallback_after_coverage_failure
        ),
        director_feedback_mode="control_plane",
    )


def _receipts() -> tuple[dict[str, object], ...]:
    memory_ids = [row[0] for row in _ROWS]
    search = {
        "tool_id": TRIVIAQA_QA_MEMORY_TOOL_ID,
        "tool_version": "qa-memory-index-v1",
        "request": {
            "action": "search",
            "arguments": {"query": "author novel", "limit": 3},
        },
        "result": {
            "completed": True,
            "value": {
                "operation": "search",
                "query": "author novel",
                "top_k": 3,
                "memory_ids": memory_ids,
                "hits": [
                    {
                        "memory_id": memory_id,
                        "rank": rank,
                        "similarity": similarity,
                    }
                    for rank, (
                        memory_id,
                        _,
                        _,
                        similarity,
                    ) in enumerate(_ROWS, start=1)
                ],
            },
        },
        "error_type": None,
    }
    reads = tuple(
        {
            "tool_id": TRIVIAQA_QA_MEMORY_TOOL_ID,
            "tool_version": "qa-memory-index-v1",
            "request": {
                "action": "read",
                "arguments": {"memory_id": memory_id},
            },
            "result": {
                "completed": True,
                "value": {
                    "operation": "read",
                    "memory_id": memory_id,
                    "memory": {
                        "memory_id": memory_id,
                        "source_train_task_id": source_id,
                        "paraphrase_question": f"Candidate question {rank}",
                        "paraphrase_answer_statement": (
                            f"The answer is {answer}"
                        ),
                        "canonical_answer": answer,
                        "text": (
                            f"Question: Candidate question {rank}\n"
                            f"Answer: The answer is {answer}"
                        ),
                    },
                },
            },
            "error_type": None,
        }
        for rank, (memory_id, source_id, answer, _) in enumerate(
            _ROWS,
            start=1,
        )
    )
    return (search, *reads)


def _artifact(
    receipts: tuple[dict[str, object], ...],
    *,
    retrieval_status: str | None = None,
) -> str:
    selection: dict[str, object] = {
        "memory_ids": [row[0] for row in _ROWS],
    }
    if retrieval_status is not None:
        selection.update(
            {
                "retrieval_status": retrieval_status,
                "relevant_memory_ids": (
                    [_ROWS[0][0]]
                    if retrieval_status == "evidence_found"
                    else []
                ),
            }
        )
    projected, issue = (
        QARetrievalReactExecutionAdapter._qa_memory_completion_receipt_projection(
            original_question=_QUESTION,
            selection_artifact=json.dumps(selection),
            tool_receipts=receipts,
            parametric_fallback_after_coverage_failure=(
                retrieval_status is not None
            ),
        )
    )
    if issue is not None or projected is None:
        raise AssertionError(issue)
    return projected


def _metadata(
    artifact: str,
    receipts: tuple[dict[str, object], ...],
    *,
    consumed_version: str = "retriever:v1",
) -> dict[str, dict[str, object]]:
    return {
        "retriever": {
            "artifact_version": "retriever:v1",
            "tool_receipts": list(receipts),
        },
        "reasoner": {
            "artifact_version": "reasoner:v1",
            "input_artifact_versions": {
                "retriever": consumed_version,
            },
            "input_artifact_provenance": [
                {
                    "source_agent_id": "retriever",
                    "target_agent_id": "reasoner",
                    "artifact": artifact,
                    "artifact_version": consumed_version,
                    "tool_receipts": list(receipts),
                }
            ],
        },
        "verifier": {
            "artifact_version": "verifier:v1",
            "input_artifact_versions": {"reasoner": "reasoner:v1"},
            "input_artifact_provenance": [],
        },
        "formatter": {
            "artifact_version": "formatter:v1",
            "input_artifact_versions": {"verifier": "verifier:v1"},
            "input_artifact_provenance": [],
        },
    }


def _execution(
    env: AgentWorkflowEnv,
    artifact: str,
    metadata: dict[str, dict[str, object]],
) -> AgentRuntimeResult:
    return AgentRuntimeResult(
        run_id="topk-env-routing",
        graph_revision=env.graph.revision,
        output_agent_id="formatter",
        final_answer="<answer>Ada</answer>",
        outputs={
            "retriever": artifact,
            "reasoner": "{}",
            "verifier": "{}",
            "formatter": "<answer>Ada</answer>",
        },
        calls=(),
        block_completion_order=(),
        output_metadata=metadata,
    )


def _fallback_execution(
    env: AgentWorkflowEnv,
    artifact: str,
    receipts: tuple[dict[str, object], ...],
    *,
    candidate: str = "Ada",
) -> AgentRuntimeResult:
    reasoner_artifact = json.dumps(
        {
            "question_scope": _QUESTION,
            "retrieval_status": "knowledge_base_coverage_failure",
            "answer_source": "parametric_knowledge",
            "answer_type": "entity",
            "answer_cardinality": "single",
            "candidate_answer": candidate,
        }
    )
    verifier_artifact = json.dumps(
        {
            "candidate_answer": candidate,
            "retrieval_status": "knowledge_base_coverage_failure",
            "answer_source": "parametric_knowledge",
            "evidence_supported": False,
            "scope_preserved": True,
            "answer_type_cardinality_correct": True,
            "minimal_answer_surface": True,
            "verification_status": "parametric_fallback",
        }
    )
    metadata = _metadata(artifact, receipts)
    metadata["verifier"]["input_artifact_provenance"] = [
        {
            "source_agent_id": "reasoner",
            "target_agent_id": "verifier",
            "artifact": reasoner_artifact,
            "artifact_version": "reasoner:v1",
            "tool_receipts": list(receipts),
        }
    ]
    return AgentRuntimeResult(
        run_id="topk-parametric-fallback",
        graph_revision=env.graph.revision,
        output_agent_id="formatter",
        final_answer=f"<answer>{candidate}</answer>",
        outputs={
            "retriever": artifact,
            "reasoner": reasoner_artifact,
            "verifier": verifier_artifact,
            "formatter": f"<answer>{candidate}</answer>",
        },
        calls=(),
        block_completion_order=(),
        output_metadata=metadata,
    )


class TriviaQATopKEnvRoutingTests(unittest.TestCase):
    def test_fallback_retriever_recovery_matches_tool_first_schema(self) -> None:
        env = _env(parametric_fallback_after_coverage_failure=True)
        values = env._triviaqa_evidence_retriever_recovery_field_values(
            "retriever"
        )
        self.assertIsNotNone(values)
        self.assertEqual(
            _QA_MEMORY_EVIDENCE_RETRIEVER_RECOVERY_COMPLETION,
            values["completion_condition"],
        )

    def test_agent_ids_are_not_numeric_answer_literals(self) -> None:
        env = _env(parametric_fallback_after_coverage_failure=True)
        issue = env._contract_obligation_issue(
            AgentAction(
                AgentActionType.MODIFY_AGENT,
                agent_id="reasoner",
                contract=(
                    "derive one semantic candidate from artifacts routed by "
                    "node_1 and node_5"
                ),
            )
        )
        self.assertIsNone(issue)

    def test_child_agent_prompts_switch_only_after_typed_coverage_failure(
        self,
    ) -> None:
        receipts = _receipts()
        artifact = _artifact(
            receipts,
            retrieval_status="knowledge_base_coverage_failure",
        )
        graph = _graph()
        model = ModelSpec("fake-model", "fake")
        provider = ProviderSpec("fake", kind="test")
        reasoner_request = AgentRequest(
            request_id="reasoner-request",
            run_id="prompt-test",
            graph_revision=graph.revision,
            problem=_QUESTION,
            agent=graph.get_node("reasoner"),
            model=model,
            provider=provider,
            phase=ExecutionPhase.SINGLE,
            semantic_protocol=QA_VERIFIED_ANSWER_LINEAGE_PROTOCOL,
            upstream=(
                UpstreamMessage(
                    "retriever",
                    "reasoner",
                    artifact,
                    # FlowSteer's production Canvas defaults AgentNode
                    # artifact_type to text. Exact schema, current artifact
                    # version, and Tool receipts establish this wire's type.
                    artifact_type="text",
                    tool_receipts=receipts,
                    artifact_version="retriever:v1",
                ),
            ),
        )
        reasoner_prompt = "\n".join(
            message["content"]
            for message in build_agent_messages(reasoner_request)
        )
        self.assertIn("parametric knowledge", reasoner_prompt)
        self.assertIn("mandatory local QA-memory search", reasoner_prompt)
        self.assertIn('answer_type exactly to "entity"', reasoner_prompt)
        self.assertIn('answer_cardinality exactly to "single"', reasoner_prompt)

        reasoner_revision_request = AgentRequest(
            request_id="reasoner-revision-request",
            run_id="prompt-test",
            graph_revision=graph.revision,
            problem=_QUESTION,
            agent=graph.get_node("reasoner"),
            model=model,
            provider=provider,
            phase=ExecutionPhase.REVISION,
            semantic_protocol=QA_VERIFIED_ANSWER_LINEAGE_PROTOCOL,
            own_draft=json.dumps(
                {
                    "question_scope": _QUESTION,
                    "retrieval_status": "knowledge_base_coverage_failure",
                    "answer_source": "parametric_knowledge",
                    "answer_type": "entity",
                    "answer_cardinality": "single",
                    "candidate_answer": "Ada",
                }
            ),
            peer_draft=UpstreamMessage(
                "retriever",
                "reasoner",
                artifact,
                artifact_type="text",
                tool_receipts=receipts,
                artifact_version="retriever:revision:v1",
            ),
        )
        revision_reasoner_prompt = "\n".join(
            message["content"]
            for message in build_agent_messages(reasoner_revision_request)
        )
        self.assertIn("parametric knowledge", revision_reasoner_prompt)
        self.assertIn("mandatory local QA-memory search", revision_reasoner_prompt)

        reasoner_artifact = json.dumps(
            {
                "question_scope": _QUESTION,
                "retrieval_status": "knowledge_base_coverage_failure",
                "answer_source": "parametric_knowledge",
                "answer_type": "entity",
                "answer_cardinality": "single",
                "candidate_answer": "Ada",
            }
        )
        verifier_request = AgentRequest(
            request_id="verifier-request",
            run_id="prompt-test",
            graph_revision=graph.revision,
            problem=_QUESTION,
            agent=graph.get_node("verifier"),
            model=model,
            provider=provider,
            phase=ExecutionPhase.SINGLE,
            semantic_protocol=QA_VERIFIED_ANSWER_LINEAGE_PROTOCOL,
            upstream=(
                UpstreamMessage(
                    "reasoner",
                    "verifier",
                    reasoner_artifact,
                    artifact_type="reasoning_artifact",
                    tool_receipts=receipts,
                    artifact_version="reasoner:v1",
                ),
            ),
        )
        verifier_prompt = "\n".join(
            message["content"]
            for message in build_agent_messages(verifier_request)
        )
        self.assertIn("evidence_supported to false", verifier_prompt)
        self.assertIn("parametric_fallback", verifier_prompt)

        verifier_revision_request = AgentRequest(
            request_id="verifier-revision-request",
            run_id="prompt-test",
            graph_revision=graph.revision,
            problem=_QUESTION,
            agent=graph.get_node("verifier"),
            model=model,
            provider=provider,
            phase=ExecutionPhase.REVISION,
            semantic_protocol=QA_VERIFIED_ANSWER_LINEAGE_PROTOCOL,
            own_draft=json.dumps(
                {
                    "question_scope": _QUESTION,
                    "retrieval_status": "knowledge_base_coverage_failure",
                    "answer_source": "parametric_knowledge",
                    "candidate_answer": "Ada",
                    "evidence_supported": False,
                    "entity_relation_consistent": True,
                    "scope_preserved": True,
                    "answer_type_cardinality_correct": True,
                    "verification_status": "parametric_fallback",
                }
            ),
            peer_draft=UpstreamMessage(
                "reasoner",
                "verifier",
                reasoner_artifact,
                artifact_type="reasoning_artifact",
                tool_receipts=receipts,
                artifact_version="reasoner:revision:v1",
            ),
        )
        revision_verifier_prompt = "\n".join(
            message["content"]
            for message in build_agent_messages(verifier_revision_request)
        )
        self.assertIn("evidence_supported to false", revision_verifier_prompt)
        self.assertIn("parametric_fallback", revision_verifier_prompt)

        untyped_request = AgentRequest(
            request_id="untyped-reasoner-request",
            run_id="prompt-test",
            graph_revision=graph.revision,
            problem=_QUESTION,
            agent=graph.get_node("reasoner"),
            model=model,
            provider=provider,
            phase=ExecutionPhase.SINGLE,
            semantic_protocol=QA_VERIFIED_ANSWER_LINEAGE_PROTOCOL,
            upstream=(
                UpstreamMessage(
                    "untyped",
                    "reasoner",
                    artifact,
                    tool_receipts=(),
                    artifact_version="retriever:v1",
                ),
            ),
        )
        untyped_prompt = "\n".join(
            message["content"]
            for message in build_agent_messages(untyped_request)
        )
        self.assertNotIn("mandatory local QA-memory search", untyped_prompt)

        evidence_receipts = json.loads(json.dumps(receipts))
        evidence_memory = evidence_receipts[1]["result"]["value"]["memory"]
        evidence_memory["paraphrase_question"] = _QUESTION
        evidence_memory["paraphrase_answer_statement"] = (
            "Ada is the author who wrote the novel."
        )
        evidence_memory["text"] = (
            f"Question: {_QUESTION}\n"
            "Answer: Ada is the author who wrote the novel."
        )
        evidence_artifact = _artifact(
            tuple(evidence_receipts),
            retrieval_status="evidence_found",
        )
        mixed_request = AgentRequest(
            request_id="mixed-retriever-status-request",
            run_id="prompt-test",
            graph_revision=graph.revision,
            problem=_QUESTION,
            agent=graph.get_node("reasoner"),
            model=model,
            provider=provider,
            phase=ExecutionPhase.SINGLE,
            semantic_protocol=QA_VERIFIED_ANSWER_LINEAGE_PROTOCOL,
            upstream=(
                UpstreamMessage(
                    "coverage-retriever",
                    "reasoner",
                    artifact,
                    tool_receipts=receipts,
                    artifact_version="coverage:v1",
                ),
                UpstreamMessage(
                    "evidence-retriever",
                    "reasoner",
                    evidence_artifact,
                    tool_receipts=tuple(evidence_receipts),
                    artifact_version="evidence:v1",
                ),
            ),
        )
        mixed_prompt = "\n".join(
            message["content"]
            for message in build_agent_messages(mixed_request)
        )
        self.assertNotIn("mandatory local QA-memory search", mixed_prompt)

    def test_matching_paired_qa_blocks_coverage_failure_status(self) -> None:
        receipts = json.loads(json.dumps(_receipts()))
        first_read = receipts[1]["result"]["value"]["memory"]
        first_read["paraphrase_question"] = _QUESTION
        first_read["paraphrase_answer_statement"] = (
            "Ada is the author who wrote the novel."
        )
        first_read["text"] = (
            f"Question: {_QUESTION}\n"
            "Answer: Ada is the author who wrote the novel."
        )
        projected, issue = (
            QARetrievalReactExecutionAdapter._qa_memory_completion_receipt_projection(
                original_question=_QUESTION,
                selection_artifact=json.dumps(
                    {
                        "memory_ids": [row[0] for row in _ROWS],
                        "retrieval_status": (
                            "knowledge_base_coverage_failure"
                        ),
                        "relevant_memory_ids": [],
                    }
                ),
                tool_receipts=tuple(receipts),
                parametric_fallback_after_coverage_failure=True,
            )
        )
        self.assertIsNone(projected)
        self.assertIsNotNone(issue)
        self.assertIn("entity/relation-compatible", issue)

    def test_irrelevant_paired_qa_cannot_claim_evidence_found(self) -> None:
        projected, issue = (
            QARetrievalReactExecutionAdapter._qa_memory_completion_receipt_projection(
                original_question=(
                    "Which American-born Sinclair won the Nobel Prize for "
                    "Literature in 1930?"
                ),
                selection_artifact=json.dumps(
                    {
                        "memory_ids": [row[0] for row in _ROWS],
                        "retrieval_status": "evidence_found",
                        "relevant_memory_ids": [_ROWS[0][0]],
                    }
                ),
                tool_receipts=_receipts(),
                parametric_fallback_after_coverage_failure=True,
            )
        )
        self.assertIsNone(projected)
        self.assertIsNotNone(issue)
        self.assertIn("evidence_found is inadmissible", issue)

    def test_possessive_public_entity_anchor_admits_possessor_query(self) -> None:
        question = "What was Prince's last No 1 of the 80s?"
        entity_anchor = _question_entity_anchor_tokens(question)
        self.assertEqual(("prince's",), entity_anchor)
        self.assertTrue(
            _surface_binds_entity_anchor(
                "Prince last No 1 80s",
                entity_anchor,
            )
        )

    def test_false_evidence_found_repair_schema_forces_coverage_failure(
        self,
    ) -> None:
        class _Index:
            manifest = SimpleNamespace(
                corpus_name="triviaqa-frozen-train-qa-memory",
                corpus_version="qa-memory-test-corpus",
                index_id="qa-memory-test-index",
                format="flowsteer.triviaqa.qa-memory-embedding-index.v1",
                retrieval_backend="test",
                tool_id=TRIVIAQA_QA_MEMORY_TOOL_ID,
                frozen_top_k=3,
            )

            def search(self, query: str, *, limit: int) -> tuple[object, ...]:
                del query, limit
                return ()

            def read(self, passage_id: str) -> object:
                raise AssertionError(passage_id)

            def close(self) -> None:
                return None

        adapter = QARetrievalReactExecutionAdapter(
            gateway=_UnusedGateway(),
            tool_registry=build_qa_tool_registry(
                _Index(),
                dataset_scope=("triviaqa",),
                tool_id=TRIVIAQA_QA_MEMORY_TOOL_ID,
            ),
            max_turns=7,
            max_tool_calls=6,
            task_type="factual_qa",
            completion_policy="required_evidence",
            parametric_fallback_after_coverage_failure=True,
            retrieval_tool_id=TRIVIAQA_QA_MEMORY_TOOL_ID,
        )
        request = AgentRequest(
            request_id="retriever-repair-request",
            run_id="prompt-test",
            graph_revision=1,
            problem="Which country does the airline LACSA come from?",
            agent=AgentNode(
                "retriever",
                "fake-model",
                "retrieve the matching Q-A memory",
                role_family="evidence_retriever",
                execution_mode="react",
                allowed_tools=(TRIVIAQA_QA_MEMORY_TOOL_ID,),
            ),
            model=ModelSpec("fake-model", "fake"),
            provider=ProviderSpec("fake", kind="test"),
            phase=ExecutionPhase.SINGLE,
            semantic_protocol=QA_VERIFIED_ANSWER_LINEAGE_PROTOCOL,
        )
        observations = []
        for receipt in _receipts():
            receipt_request = receipt["request"]
            receipt_value = json.loads(
                json.dumps(receipt["result"]["value"])
            )
            if receipt_request["action"] == "search":
                receipt_value["passage_ids"] = list(
                    receipt_value["memory_ids"]
                )
                for hit in receipt_value["hits"]:
                    hit["passage_id"] = hit["memory_id"]
            else:
                receipt_value["passage_id"] = receipt_value["memory_id"]
                receipt_value["passage"] = dict(receipt_value["memory"])
            observations.append(
                {
                    "observation_status": "success",
                    "executed_action": {
                        "kind": "tool",
                        "name": receipt_request["action"],
                        "resource_id": TRIVIAQA_QA_MEMORY_TOOL_ID,
                        "arguments": receipt_request["arguments"],
                    },
                    "result": receipt_value,
                }
            )
        observations.append(
            {
                "observation_status": "schema_invalid",
                "public_error_code": (
                    "qa_semantic_artifact_invalid: TriviaQA QA-memory "
                    "evidence_found is inadmissible because the selected record "
                    "does not preserve the public question's named entity/scope "
                    "and target relation"
                ),
            }
        )

        schema = adapter._state_conditioned_response_schema(
            request,
            observations,
        )
        self.assertIsNotNone(schema)
        assert schema is not None
        value_properties = schema["properties"]["arguments"]["properties"][
            "value"
        ]["properties"]
        self.assertEqual(
            {"const": "knowledge_base_coverage_failure"},
            value_properties["retrieval_status"],
        )
        self.assertEqual(
            {"type": "array", "maxItems": 0},
            value_properties["relevant_memory_ids"],
        )

    def test_tool_first_coverage_failure_admits_parametric_child_reasoner(
        self,
    ) -> None:
        env = _env(parametric_fallback_after_coverage_failure=True)
        receipts = _receipts()
        artifact = _artifact(
            receipts,
            retrieval_status="knowledge_base_coverage_failure",
        )
        execution = _fallback_execution(env, artifact, receipts)

        self.assertIsNone(env._semantic_protocol_issue(execution))
        env._progressive_outputs = dict(execution.outputs)
        env._progressive_output_metadata = {
            agent_id: dict(metadata)
            for agent_id, metadata in execution.output_metadata.items()
        }
        self.assertEqual(
            ("reasoner", "verifier", "formatter"),
            env._active_semantic_lineage_ids(),
        )

    def test_parametric_fallback_cannot_bypass_complete_top_k_or_status(
        self,
    ) -> None:
        env = _env(parametric_fallback_after_coverage_failure=True)
        receipts = _receipts()
        coverage_artifact = _artifact(
            receipts,
            retrieval_status="knowledge_base_coverage_failure",
        )
        missing_read = receipts[:-1]
        incomplete = _fallback_execution(
            env,
            coverage_artifact,
            missing_read,
        )
        issue = env._semantic_protocol_issue(incomplete)
        self.assertIsNotNone(issue)
        self.assertIn("read(memory_id)", issue)

        evidence_receipts = json.loads(json.dumps(receipts))
        evidence_memory = evidence_receipts[1]["result"]["value"]["memory"]
        evidence_memory["paraphrase_question"] = _QUESTION
        evidence_memory["paraphrase_answer_statement"] = (
            "Ada is the author who wrote the novel."
        )
        evidence_memory["text"] = (
            f"Question: {_QUESTION}\n"
            "Answer: Ada is the author who wrote the novel."
        )
        evidence_artifact = _artifact(
            tuple(evidence_receipts),
            retrieval_status="evidence_found",
        )
        evidence_execution = _fallback_execution(
            env,
            evidence_artifact,
            tuple(evidence_receipts),
        )
        issue = env._semantic_protocol_issue(evidence_execution)
        self.assertIsNotNone(issue)
        self.assertIn("Reasoner", issue)

    def test_v13_default_does_not_admit_coverage_fallback(self) -> None:
        env = _env()
        receipts = _receipts()
        coverage_artifact = _artifact(
            receipts,
            retrieval_status="knowledge_base_coverage_failure",
        )
        issue = env._semantic_protocol_issue(
            _fallback_execution(env, coverage_artifact, receipts)
        )
        self.assertIsNotNone(issue)

    def test_complete_top_k_artifact_is_routed_on_explicit_direct_relation(
        self,
    ) -> None:
        env = _env()
        receipts = _receipts()
        artifact = _artifact(receipts)
        metadata = _metadata(artifact, receipts)

        self.assertEqual(
            [row[0] for row in _ROWS],
            [
                candidate["memory_id"]
                for candidate in json.loads(artifact)["candidates"]
            ],
        )
        self.assertTrue(
            env.graph.relation_bits(
                "retriever",
                "reasoner",
            ).source_to_target
        )
        self.assertIsNone(
            env._triviaqa_qa_memory_ingress_issue(
                {"retriever": artifact},
                metadata,
                retriever_id="retriever",
                reasoner_id="reasoner",
            )
        )

        env._graph.set_relation("retriever", "reasoner", False, False)
        issue = env._triviaqa_qa_memory_ingress_issue(
            {"retriever": artifact},
            metadata,
            retriever_id="retriever",
            reasoner_id="reasoner",
        )
        self.assertIsNotNone(issue)
        self.assertIn("explicit direct", issue)

    def test_missing_candidate_cannot_finish_or_replace_retriever(self) -> None:
        env = _env()
        receipts = _receipts()
        artifact_fields = json.loads(_artifact(receipts))
        artifact_fields["candidates"].pop()
        artifact = json.dumps(artifact_fields)
        metadata = _metadata(artifact, receipts)

        issue = env._semantic_protocol_issue(
            _execution(env, artifact, metadata)
        )
        self.assertIsNotNone(issue)
        self.assertIn("complete embedding-ranked top-k", issue)
        env._progressive_outputs["retriever"] = artifact
        env._progressive_output_metadata["retriever"] = metadata["retriever"]
        self.assertFalse(
            env._semantic_replacement_has_valid_artifact(
                "retriever",
                "evidence_retriever",
            )
        )

    def test_missing_read_cannot_finish_or_replace_retriever(self) -> None:
        env = _env()
        complete_receipts = _receipts()
        artifact = _artifact(complete_receipts)
        incomplete_receipts = complete_receipts[:-1]
        metadata = _metadata(artifact, incomplete_receipts)

        issue = env._semantic_protocol_issue(
            _execution(env, artifact, metadata)
        )
        self.assertIsNotNone(issue)
        self.assertIn("read(memory_id)", issue)
        env._progressive_outputs["retriever"] = artifact
        env._progressive_output_metadata["retriever"] = metadata["retriever"]
        self.assertFalse(
            env._semantic_replacement_has_valid_artifact(
                "retriever",
                "evidence_retriever",
            )
        )

    def test_stale_retriever_version_cannot_finish_or_replace_reasoner(
        self,
    ) -> None:
        env = _env()
        receipts = _receipts()
        artifact = _artifact(receipts)
        metadata = _metadata(
            artifact,
            receipts,
            consumed_version="retriever:stale",
        )

        issue = env._semantic_protocol_issue(
            _execution(env, artifact, metadata)
        )
        self.assertIsNotNone(issue)
        self.assertIn("not bound to the current artifact version", issue)
        env._progressive_outputs.update(
            {
                "retriever": artifact,
                "reasoner": "{}",
            }
        )
        env._progressive_output_metadata.update(metadata)
        self.assertFalse(
            env._semantic_replacement_has_valid_artifact(
                "reasoner",
                "reasoner",
            )
        )


if __name__ == "__main__":
    unittest.main()
