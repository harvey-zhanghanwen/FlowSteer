from __future__ import annotations

import json
import unittest

from src.interactive.agent_graph import AgentGraph, AgentNode, AgentRelation
from src.interactive.agent_runtime import AgentRuntime, AgentRuntimeResult
from src.interactive.agent_workflow_env import (
    AgentWorkflowEnv,
    _HOTPOTQA_FORMAT_CONTRACT,
)
from src.interactive.model_registry import ModelRegistry, ModelSpec, ProviderSpec
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


def _env() -> AgentWorkflowEnv:
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


def _artifact(receipts: tuple[dict[str, object], ...]) -> str:
    projected, issue = (
        QARetrievalReactExecutionAdapter._qa_memory_completion_receipt_projection(
            original_question=_QUESTION,
            selection_artifact=json.dumps(
                {"memory_ids": [row[0] for row in _ROWS]}
            ),
            tool_receipts=receipts,
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


class TriviaQATopKEnvRoutingTests(unittest.TestCase):
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
