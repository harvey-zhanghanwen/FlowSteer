from __future__ import annotations

from dataclasses import dataclass, replace
import json
from types import SimpleNamespace
import unittest

from src.interactive.agent_graph import AgentGraph, AgentNode, AgentRelation
from src.interactive.agent_runtime import (
    AgentRequest,
    ExecutionPhase,
    UpstreamMessage,
)
from src.interactive.agent_workflow_env import AgentWorkflowEnv
from src.interactive.model_registry import ModelSpec, ProviderSpec
from src.interactive.openai_gateway import build_agent_messages
from src.interactive.qa_tool_adapter import (
    QARetrievalReactExecutionAdapter,
    QA_VERIFIED_ANSWER_LINEAGE_PROTOCOL,
    TRIVIAQA_QA_MEMORY_TOOL_ID,
    build_qa_tool_registry,
    qa_reasoner_artifact_json_schema,
)
from src.interactive.tool_runtime import ToolRequest


@dataclass(frozen=True)
class FactManifest:
    corpus_name: str = "triviaqa-semantic-fact-memory"
    corpus_version: str = "v1"
    index_id: str = "triviaqa-fact-memory-v1"
    format: str = "flowsteer.triviaqa.fact-memory-embedding-index.v1"
    retrieval_backend: str = "bge-cosine"
    tool_id: str = TRIVIAQA_QA_MEMORY_TOOL_ID
    record_kind: str = "fact_memory"
    frozen_top_k: int = 2


@dataclass(frozen=True)
class FactHit:
    memory_id: str
    rank: int
    similarity: float
    fact_text: str


@dataclass(frozen=True)
class FactRecord:
    schema_version: str
    memory_id: str
    tool_id: str
    fact_text: str


class FactIndex:
    manifest = FactManifest()

    def __init__(self) -> None:
        self.records = {
            "fact-1": FactRecord(
                "flowsteer.triviaqa.fact_memory.record.v1",
                "fact-1",
                TRIVIAQA_QA_MEMORY_TOOL_ID,
                "Ada Lovelace wrote the first published algorithm.",
            ),
            "fact-2": FactRecord(
                "flowsteer.triviaqa.fact_memory.record.v1",
                "fact-2",
                TRIVIAQA_QA_MEMORY_TOOL_ID,
                "Charles Babbage designed the Analytical Engine.",
            ),
        }

    def search(self, query: str, *, limit: int) -> tuple[FactHit, ...]:
        del query
        return tuple(
            FactHit(record.memory_id, rank, 1.0 - rank / 10, record.fact_text)
            for rank, record in enumerate(self.records.values(), start=1)
        )[:limit]

    def read(self, memory_id: str) -> FactRecord:
        return self.records[memory_id]

    def close(self) -> None:
        return None


def _fact_receipts() -> tuple[dict[str, object], ...]:
    hits = [
        {
            "memory_id": "fact-1",
            "rank": 1,
            "similarity": 0.9,
            "fact_text": "Ada Lovelace wrote the first published algorithm.",
        },
        {
            "memory_id": "fact-2",
            "rank": 2,
            "similarity": 0.8,
            "fact_text": "Charles Babbage designed the Analytical Engine.",
        },
    ]
    search = {
        "tool_id": TRIVIAQA_QA_MEMORY_TOOL_ID,
        "error_type": None,
        "request": {
            "action": "search",
            "arguments": {"query": "first published algorithm", "limit": 2},
        },
        "result": {
            "completed": True,
            "value": {
                "operation": "search",
                "query": "first published algorithm",
                "top_k": 2,
                "memory_ids": ["fact-1", "fact-2"],
                "hits": hits,
            },
        },
    }
    reads = tuple(
        {
            "tool_id": TRIVIAQA_QA_MEMORY_TOOL_ID,
            "error_type": None,
            "request": {
                "action": "read",
                "arguments": {"memory_id": hit["memory_id"]},
            },
            "result": {
                "completed": True,
                "value": {
                    "operation": "read",
                    "memory_id": hit["memory_id"],
                    "memory": {
                        "memory_id": hit["memory_id"],
                        "fact_text": hit["fact_text"],
                    },
                },
            },
        }
        for hit in hits
    )
    return (search, *reads)


def _reasoner_request(artifact: str, receipts: tuple[dict[str, object], ...]) -> AgentRequest:
    return AgentRequest(
        request_id="triviaqa:fact-memory",
        run_id="fact-memory",
        graph_revision=2,
        problem="Who wrote the first published algorithm?",
        agent=AgentNode(
            "reasoner",
            "model",
            "derive the semantic answer from routed evidence",
            role_family="reasoner",
        ),
        model=ModelSpec("model", "provider"),
        provider=ProviderSpec("provider", kind="test"),
        phase=ExecutionPhase.SINGLE,
        semantic_protocol=QA_VERIFIED_ANSWER_LINEAGE_PROTOCOL,
        upstream=(
            UpstreamMessage(
                "retriever",
                "reasoner",
                artifact,
                graph_revision=2,
                artifact_type="evidence",
                artifact_version="artifact-v1",
                tool_receipts=receipts,
            ),
        ),
    )


class TriviaQAFactMemoryRuntimeTests(unittest.IsolatedAsyncioTestCase):
    async def test_registry_projects_fact_only_search_and_read_payloads(self) -> None:
        registry = build_qa_tool_registry(FactIndex())
        self.assertEqual((TRIVIAQA_QA_MEMORY_TOOL_ID,), registry.resource_ids)
        capability = registry.require_capability(TRIVIAQA_QA_MEMORY_TOOL_ID)
        self.assertEqual(
            "fact_memory",
            capability.output_schema["x-flowsteer-record-kind"],
        )

        result, receipt = await registry.ainvoke_with_receipt(
            TRIVIAQA_QA_MEMORY_TOOL_ID,
            ToolRequest(
                "search",
                {"query": "first published algorithm", "limit": 2},
            ),
        )
        self.assertIsNone(receipt.error_type)
        assert result is not None
        self.assertEqual(["fact-1", "fact-2"], result.value["memory_ids"])
        self.assertNotIn("passage_ids", result.value)
        self.assertEqual(
            {"memory_id", "rank", "similarity", "fact_text"},
            set(result.value["hits"][0]),
        )

        result, receipt = await registry.ainvoke_with_receipt(
            TRIVIAQA_QA_MEMORY_TOOL_ID,
            ToolRequest("read", {"memory_id": "fact-1"}),
        )
        self.assertIsNone(receipt.error_type)
        assert result is not None
        self.assertEqual(
            {"memory_id", "fact_text"},
            set(result.value["memory"]),
        )
        self.assertNotIn("canonical_answer", json.dumps(result.value))
        self.assertNotIn("source_train_task_id", json.dumps(result.value))

    def test_fact_projection_requires_rank_ordered_read_for_every_topk_hit(self) -> None:
        receipts = _fact_receipts()
        selection = json.dumps(
            {
                "memory_ids": ["fact-1", "fact-2"],
                "retrieval_status": "evidence_found",
                "relevant_memory_ids": ["fact-1"],
            }
        )
        projected, issue = (
            QARetrievalReactExecutionAdapter._qa_memory_completion_receipt_projection(
                original_question="Who wrote the first published algorithm?",
                selection_artifact=selection,
                tool_receipts=receipts,
                parametric_fallback_after_coverage_failure=True,
                expected_source_task_id="must-not-enter-fact-profile",
            )
        )
        self.assertIsNone(issue)
        assert projected is not None
        artifact = json.loads(projected)
        self.assertEqual(["fact-1", "fact-2"], [c["memory_id"] for c in artifact["candidates"]])
        self.assertTrue(all(set(c) == {"memory_id", "rank", "similarity", "fact_text"} for c in artifact["candidates"]))
        self.assertNotIn("canonical_answer", projected)
        self.assertNotIn("source_train_task_id", projected)

        _, issue = QARetrievalReactExecutionAdapter._qa_memory_completion_receipt_projection(
            original_question="Who wrote the first published algorithm?",
            selection_artifact=selection,
            tool_receipts=(receipts[0], receipts[2], receipts[1]),
            parametric_fallback_after_coverage_failure=True,
        )
        self.assertIn("original rank order", issue or "")

    def test_ranked_action_domain_reads_every_fact_before_completion(self) -> None:
        registry = build_qa_tool_registry(FactIndex())
        adapter = QARetrievalReactExecutionAdapter(
            gateway=SimpleNamespace(generate=lambda request: None),
            tool_registry=registry,
            max_turns=8,
            max_tool_calls=4,
            task_type="factual_qa",
            completion_policy="required_evidence",
            parametric_fallback_after_coverage_failure=True,
            retrieval_tool_id=TRIVIAQA_QA_MEMORY_TOOL_ID,
        )
        request = AgentRequest(
            request_id="triviaqa:worker",
            run_id="fact-memory",
            graph_revision=1,
            problem="Who wrote the first published algorithm?",
            agent=AgentNode(
                "retriever",
                "model",
                "retrieve evidence",
                role_family="evidence_retriever",
                allowed_tools=(TRIVIAQA_QA_MEMORY_TOOL_ID,),
                execution_mode="react",
            ),
            model=ModelSpec("model", "provider"),
            provider=ProviderSpec("provider", kind="test"),
            phase=ExecutionPhase.SINGLE,
            semantic_protocol=QA_VERIFIED_ANSWER_LINEAGE_PROTOCOL,
        )
        receipts = _fact_receipts()
        observations = []
        for receipt in receipts:
            observations.append(
                {
                    "observation_status": "success",
                    "executed_action": {
                        "kind": "tool",
                        "name": receipt["request"]["action"],
                        "resource_id": TRIVIAQA_QA_MEMORY_TOOL_ID,
                        "arguments": receipt["request"]["arguments"],
                    },
                    "result": receipt["result"]["value"],
                }
            )
        actions, complete = adapter._state_conditioned_action_domain(
            request,
            observations[:1],
        )
        self.assertEqual(
            frozenset({(TRIVIAQA_QA_MEMORY_TOOL_ID, "read")}),
            actions,
        )
        self.assertFalse(complete)
        _, complete = adapter._state_conditioned_action_domain(
            request,
            observations[:2],
        )
        self.assertFalse(complete)
        actions, complete = adapter._state_conditioned_action_domain(
            request,
            observations,
        )
        self.assertEqual(frozenset(), actions)
        self.assertTrue(complete)

    def test_fact_read_is_receipt_grounded_and_extra_answer_field_is_rejected(self) -> None:
        receipt = _fact_receipts()[1]
        self.assertTrue(
            AgentWorkflowEnv._successful_read_receipt(
                receipt,
                TRIVIAQA_QA_MEMORY_TOOL_ID,
            )
        )
        self.assertEqual(
            "Ada Lovelace wrote the first published algorithm.",
            AgentWorkflowEnv._successful_read_text(
                receipt,
                TRIVIAQA_QA_MEMORY_TOOL_ID,
            ),
        )
        leaked = json.loads(json.dumps(receipt))
        leaked["result"]["value"]["memory"]["canonical_answer"] = "Ada Lovelace"
        self.assertFalse(
            AgentWorkflowEnv._successful_read_receipt(
                leaked,
                TRIVIAQA_QA_MEMORY_TOOL_ID,
            )
        )

    def test_react_adapter_admits_canonical_fact_read_receipt(self) -> None:
        receipt = _fact_receipts()[1]

        self.assertTrue(
            QARetrievalReactExecutionAdapter._successful_read_receipt(
                receipt,
                TRIVIAQA_QA_MEMORY_TOOL_ID,
            )
        )
        self.assertEqual(
            "Ada Lovelace wrote the first published algorithm.",
            QARetrievalReactExecutionAdapter._successful_read_text(
                receipt,
                TRIVIAQA_QA_MEMORY_TOOL_ID,
            ),
        )

        leaked = json.loads(json.dumps(receipt))
        leaked["result"]["value"]["memory"]["canonical_answer"] = "Ada Lovelace"
        self.assertFalse(
            QARetrievalReactExecutionAdapter._successful_read_receipt(
                leaked,
                TRIVIAQA_QA_MEMORY_TOOL_ID,
            )
        )

    def test_reasoner_schema_and_prompt_have_no_canonical_answer_constraint(self) -> None:
        projected, issue = (
            QARetrievalReactExecutionAdapter._qa_memory_completion_receipt_projection(
                original_question="Who wrote the first published algorithm?",
                selection_artifact=json.dumps(
                    {
                        "memory_ids": ["fact-1", "fact-2"],
                        "retrieval_status": "evidence_found",
                        "relevant_memory_ids": ["fact-1"],
                    }
                ),
                tool_receipts=_fact_receipts(),
                parametric_fallback_after_coverage_failure=True,
            )
        )
        self.assertIsNone(issue)
        assert projected is not None
        request = _reasoner_request(projected, _fact_receipts())
        schema = qa_reasoner_artifact_json_schema(request)
        assert schema is not None
        candidate_schema = schema["properties"]["candidate_answer"]
        self.assertNotIn("const", candidate_schema)
        proposition_schema = schema["properties"]["evidence_propositions"]["items"]
        self.assertNotIn("const", proposition_schema["properties"]["subject"])
        messages = build_agent_messages(request)
        self.assertIn("self-contained declarative fact_text", messages[0]["content"])
        self.assertIn("hidden canonical answer", messages[0]["content"])

        misrouted = replace(
            request,
            upstream=(
                UpstreamMessage(
                    "retriever",
                    "different-agent",
                    projected,
                    graph_revision=2,
                    artifact_type="evidence",
                    artifact_version="artifact-v1",
                    tool_receipts=_fact_receipts(),
                ),
            ),
        )
        self.assertNotIn(
            "self-contained declarative fact_text",
            build_agent_messages(misrouted)[0]["content"],
        )

    def test_fact_artifact_ingress_requires_explicit_direct_relation(self) -> None:
        projected, issue = (
            QARetrievalReactExecutionAdapter._qa_memory_completion_receipt_projection(
                original_question="Who wrote the first published algorithm?",
                selection_artifact=json.dumps(
                    {
                        "memory_ids": ["fact-1", "fact-2"],
                        "retrieval_status": "evidence_found",
                        "relevant_memory_ids": ["fact-1"],
                    }
                ),
                tool_receipts=_fact_receipts(),
                parametric_fallback_after_coverage_failure=True,
            )
        )
        self.assertIsNone(issue)
        assert projected is not None
        retriever = AgentNode(
            "retriever",
            "model",
            "retrieve evidence",
            role_family="evidence_retriever",
            allowed_tools=(TRIVIAQA_QA_MEMORY_TOOL_ID,),
            execution_mode="react",
        )
        reasoner = AgentNode(
            "reasoner",
            "model",
            "derive answer",
            role_family="reasoner",
        )
        env = object.__new__(AgentWorkflowEnv)
        env.semantic_protocol = QA_VERIFIED_ANSWER_LINEAGE_PROTOCOL
        env.required_evidence_tool_id = TRIVIAQA_QA_MEMORY_TOOL_ID
        env.parametric_fallback_after_coverage_failure = True
        env._problem = "Who wrote the first published algorithm?"
        env._graph = AgentGraph((retriever, reasoner))
        outputs = {"retriever": projected}
        metadata = {
            "retriever": {
                "artifact_version": "artifact-v1",
                "tool_receipts": list(_fact_receipts()),
            },
            "reasoner": {
                "input_artifact_versions": {"retriever": "artifact-v1"},
                "input_artifact_provenance": [
                    {
                        "source_agent_id": "retriever",
                        "target_agent_id": "reasoner",
                        "artifact_version": "artifact-v1",
                        "artifact": projected,
                        "tool_receipts": list(_fact_receipts()),
                    }
                ],
            },
        }
        self.assertIn(
            "explicit direct",
            env._triviaqa_qa_memory_ingress_issue(
                outputs,
                metadata,
                retriever_id="retriever",
                reasoner_id="reasoner",
            )
            or "",
        )
        env._graph = AgentGraph(
            (retriever, reasoner),
            (AgentRelation("retriever", "reasoner", True, False),),
        )
        self.assertIsNone(
            env._triviaqa_qa_memory_ingress_issue(
                outputs,
                metadata,
                retriever_id="retriever",
                reasoner_id="reasoner",
            )
        )


if __name__ == "__main__":
    unittest.main()
