from __future__ import annotations

import json
import unittest

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
)
from src.interactive.model_registry import ModelRegistry, ModelSpec, ProviderSpec
from src.interactive.openai_gateway import build_agent_messages
from src.interactive.qa_tool_adapter import (
    QA_VERIFIED_ANSWER_LINEAGE_PROTOCOL,
    QARetrievalReactExecutionAdapter,
    TRIVIAQA_QA_MEMORY_TOOL_ID,
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
                    artifact_type="retrieval_evidence",
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
                    artifact_type="semantic_candidate",
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
                    tool_receipts=receipts,
                    artifact_version="retriever:v1",
                ),
            ),
        )
        untyped_prompt = "\n".join(
            message["content"]
            for message in build_agent_messages(untyped_request)
        )
        self.assertNotIn("mandatory local QA-memory search", untyped_prompt)

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

        evidence_artifact = _artifact(
            receipts,
            retrieval_status="evidence_found",
        )
        evidence_execution = _fallback_execution(
            env,
            evidence_artifact,
            receipts,
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
