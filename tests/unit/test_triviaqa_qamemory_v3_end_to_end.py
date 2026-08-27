from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
import unittest

from scripts import analyze_triviaqa_qa_memory_results as analysis
from src.interactive.agent_graph import AgentGraph, AgentNode, AgentRelation
from src.interactive.agent_runtime import (
    AgentFailureRecord,
    AgentRequest,
    AgentResponse,
    AgentRuntime,
    ExecutionPhase,
)
from src.interactive.agent_workflow_env import (
    AgentWorkflowEnv,
    _HOTPOTQA_FORMAT_CONTRACT,
)
from src.interactive.config_loader import load_yaml, validate_agent_graph_config
from src.interactive.director import (
    AgentGraphOrchestrator,
    DirectorResponse,
    decode_director_transcript,
)
from src.interactive.model_registry import ModelRegistry, ModelSpec, ProviderSpec
from src.interactive.qa_tool_adapter import (
    QARetrievalReactExecutionAdapter,
    TRIVIAQA_QA_MEMORY_TOOL_ID,
    build_qa_tool_registry,
)
from src.interactive.rollout_collector import execution_record_from_call


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "config/evaluation_triviaqa_qa_memory_unified_v3.yaml"
V4_MEMORY_PATH = PROJECT_ROOT / "data/triviaqa_qa_memory_v4/train_qa_memory.jsonl"
V4_MANIFEST_PATH = PROJECT_ROOT / "data/triviaqa_qa_memory_v4/index/manifest.json"
QUESTION = "Which British general died at Khartoum in 1885?"


def _first_jsonl_rows(
    path: Path,
    *,
    count: int,
) -> tuple[dict[str, object], ...]:
    rows: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8") as handle:
        for _ in range(count):
            value = json.loads(next(handle))
            if not isinstance(value, dict):
                raise AssertionError(
                    "expected a materialized v4 QA-memory record"
                )
            rows.append(value)
    return tuple(rows)


class _RealRowIndex:
    """Fake embedding boundary backed by three materialized v4 records."""

    def __init__(self, rows: tuple[dict[str, object], ...]) -> None:
        manifest = json.loads(V4_MANIFEST_PATH.read_text(encoding="utf-8"))
        self.manifest = SimpleNamespace(**manifest)
        self.rows = tuple(dict(row) for row in rows)
        self.rows_by_memory_id = {
            str(row["memory_id"]): row for row in self.rows
        }
        self.search_calls: list[tuple[str, int]] = []
        self.read_calls: list[str] = []

    @staticmethod
    def _record(
        row: dict[str, object],
        *,
        rank: int | None = None,
        similarity: float | None = None,
    ) -> SimpleNamespace:
        memory_id = str(row["memory_id"])
        question = str(row["paraphrase_question"])
        answer_statement = str(row["paraphrase_answer_statement"])
        values = {
            **row,
            "passage_id": memory_id,
            "document_id": str(row["source_train_task_id"]),
            "title": question,
            "snippet": question,
            "text": f"Question: {question}\nAnswer: {answer_statement}",
        }
        if rank is not None:
            values.update({"rank": rank, "similarity": similarity})
        return SimpleNamespace(**values)

    def search(self, query: str, *, limit: int) -> tuple[SimpleNamespace, ...]:
        self.search_calls.append((query, limit))
        return tuple(
            self._record(
                row,
                rank=rank,
                similarity=1.0 - (rank / 100),
            )
            for rank, row in enumerate(self.rows[:limit], start=1)
        )

    def read(self, memory_id: str) -> SimpleNamespace:
        self.read_calls.append(memory_id)
        row = self.rows_by_memory_id.get(memory_id)
        if row is None:
            raise KeyError(memory_id)
        return self._record(row)

    def close(self) -> None:
        return None


def _action(
    name: str,
    arguments: object,
    *,
    resource_id: str | None,
) -> str:
    return json.dumps(
        {
            "kind": "complete" if name == "complete" else "tool",
            "name": name,
            "resource_id": resource_id,
            "skill_id": None,
            "arguments": arguments,
        },
        ensure_ascii=False,
        sort_keys=True,
    )


class _ReactGateway:
    def __init__(self, rows: tuple[dict[str, object], ...]) -> None:
        memory_ids = [str(row["memory_id"]) for row in rows]
        self.outputs = [
            _action(
                "search",
                {"query": QUESTION, "limit": 3},
                resource_id=TRIVIAQA_QA_MEMORY_TOOL_ID,
            ),
            *(
                _action(
                    "read",
                    {"memory_id": memory_id},
                    resource_id=TRIVIAQA_QA_MEMORY_TOOL_ID,
                )
                for memory_id in memory_ids
            ),
            _action(
                "complete",
                {"value": {"memory_ids": memory_ids}},
                resource_id=None,
            ),
        ]
        self.requests: list[AgentRequest] = []

    async def generate(self, request: AgentRequest) -> AgentResponse:
        self.requests.append(request)
        if request.agent.role_family != "evidence_retriever":
            raise AssertionError("only the Evidence Retriever may execute ReAct")
        if request.agent.allowed_tools != (TRIVIAQA_QA_MEMORY_TOOL_ID,):
            raise AssertionError("QA-memory Tool ownership left the worker")
        return AgentResponse(self.outputs.pop(0))


class _ReasoningGateway:
    def __init__(self, rows: tuple[dict[str, object], ...]) -> None:
        self.rows = rows
        self.requests: list[AgentRequest] = []

    async def generate(self, request: AgentRequest) -> AgentResponse:
        self.requests.append(request)
        answer = str(self.rows[0]["canonical_answer"])
        statement = str(self.rows[0]["paraphrase_answer_statement"])
        if request.agent.id == "reasoner":
            return AgentResponse(
                json.dumps(
                    {
                        "question_scope": request.problem,
                        "answer_slot": {
                            "answer_type": "person",
                            "answer_cardinality": "single",
                            "qualifiers": [],
                            "proposition_index": 0,
                            "answer_field": "object_or_attribute_value",
                        },
                        "evidence_propositions": [
                            {
                                "subject": request.problem,
                                "relation": "has_answer",
                                "object_or_attribute_value": answer,
                                "qualifiers": [],
                                "evidence_span": statement,
                            }
                        ],
                        "multi_hop_chain": [f"question --has_answer--> {answer}"],
                        "candidate_answer": answer,
                        "evidence": [statement],
                    },
                    ensure_ascii=False,
                )
            )
        if request.agent.id == "verifier":
            return AgentResponse(
                json.dumps(
                    {
                        "candidate_answer": answer,
                        "evidence_supported": True,
                        "entity_attribute_binding_correct": True,
                        "alias_binding_correct": True,
                        "answer_type_cardinality_correct": True,
                        "multi_hop_complete": True,
                        "minimal_answer_surface": True,
                        "scope_preserved": True,
                        "verification_status": "supported",
                    },
                    ensure_ascii=False,
                )
            )
        if request.agent.id == "formatter":
            return AgentResponse(f"<answer>{answer}</answer>")
        raise AssertionError(f"unexpected reasoning Agent {request.agent.id!r}")


class _UnusedDirector:
    async def propose(self, prompt: str, **_: object) -> DirectorResponse:
        raise AssertionError(f"unexpected Director generation: {prompt[:40]}")


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
                    "ground answer-free evidence for the original entity and "
                    "requested relation in matching successful read Tool receipts"
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
                    "relation binding, and answer lineage without changing the candidate"
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


def _observation(prompt: str) -> dict[str, object]:
    messages = decode_director_transcript(prompt)
    if messages is None:
        raise AssertionError("expected a canonical Director transcript")
    _, separator, encoded = messages[-1]["content"].partition("\n\n")
    if not separator:
        raise AssertionError("missing current Canvas observation")
    value = json.loads(encoded)
    if not isinstance(value, dict):
        raise AssertionError("Canvas observation must be an object")
    return value


class TriviaQAQAMemoryV3EndToEndTests(unittest.IsolatedAsyncioTestCase):
    def test_nested_yaml_semantic_artifact_preserves_top_level_contract(self) -> None:
        fields, issue = AgentWorkflowEnv._structured_semantic_fields(
            """question_scope: Which person won the prize?
answer_slot:
  answer_type: entity
  target_relation: won
evidence_propositions:
  - The receipt states that Ada won the prize.
multi_hop_chain:
  - Bind the receipt subject to the requested person.
candidate_answer: Ada
evidence:
  passage_id: memory-1
""",
            (
                "question_scope",
                "answer_slot",
                "evidence_propositions",
                "multi_hop_chain",
                "candidate_answer",
                "evidence",
            ),
        )

        self.assertIsNone(issue)
        assert fields is not None
        self.assertEqual("entity", fields["answer_slot"]["answer_type"])
        self.assertEqual("Ada", fields["candidate_answer"])

    async def test_real_v4_row_worker_search_read_relation_and_lineage(self) -> None:
        config = load_yaml(CONFIG_PATH)
        validate_agent_graph_config(config)
        self.assertEqual(
            "data/triviaqa_qa_memory_v4/index",
            config["qa_tool_runtime"]["index_path"],
        )
        self.assertEqual(
            TRIVIAQA_QA_MEMORY_TOOL_ID,
            config["agent_graph"]["required_evidence_tool_id"],
        )

        rows = _first_jsonl_rows(V4_MEMORY_PATH, count=3)
        memory_ids = [str(row["memory_id"]) for row in rows]
        index = _RealRowIndex(rows)
        tool_registry = build_qa_tool_registry(index)
        react_gateway = _ReactGateway(rows)
        reasoning_gateway = _ReasoningGateway(rows)
        react_adapter = QARetrievalReactExecutionAdapter(
            gateway=react_gateway,
            tool_registry=tool_registry,
            max_turns=config["qa_tool_runtime"]["max_turns_per_agent_call"],
            max_tool_calls=config["qa_tool_runtime"]["max_tool_calls_per_agent_call"],
            task_type=None,
            completion_policy=config["qa_tool_runtime"]["completion_policy"],
            retrieval_tool_id=TRIVIAQA_QA_MEMORY_TOOL_ID,
        )
        registry = _registry()
        runtime = AgentRuntime(
            registry,
            reasoning_gateway,
            execution_adapters={"react": react_adapter},
            tool_registry=tool_registry,
            dataset_id="triviaqa",
            semantic_protocol="qa_verified_answer_lineage_v2",
        )
        graph = _graph()
        env = AgentWorkflowEnv(
            registry,
            runtime=runtime,
            graph=graph,
            problem=QUESTION,
            require_exact_answer_tag=True,
            require_format_agent=True,
            semantic_protocol="qa_verified_answer_lineage_v2",
            recovery_policy="preserve_diagnose_repair_augment",
            required_evidence_tool_id=TRIVIAQA_QA_MEMORY_TOOL_ID,
            director_feedback_mode="control_plane",
        )
        self.assertIsNone(env._semantic_edit_issue_for(graph))

        orchestrator = AgentGraphOrchestrator(
            registry,
            _UnusedDirector(),
            tool_registry=tool_registry,
            prompt_version=config["experiment"]["prompt_version"],
            semantic_protocol="qa_verified_answer_lineage_v2",
            recovery_policy="preserve_diagnose_repair_augment",
        )
        director_prompt = orchestrator.build_prompt(env, 0, ())
        observation = _observation(director_prompt)
        self.assertEqual(
            {"allowed_tools": [], "tool_calls_enabled": False},
            observation["director_execution_profile"],
        )
        reasoner_profile = observation["action_target_domains"]["add_subgraph"][
            "role_constraints"
        ]["reasoner"]
        self.assertEqual(["reasoning"], reasoner_profile["execution_modes"])
        self.assertEqual([[]], reasoner_profile["allowed_tools"])
        for row in rows:
            for private_value in (
                str(row["memory_id"]),
                str(row["source_train_task_id"]),
                str(row["paraphrase_answer_statement"]),
                str(row["canonical_answer"]),
            ):
                self.assertNotIn(private_value, director_prompt)

        result = await runtime.execute(
            graph,
            QUESTION,
            run_id="triviaqa-qamemory-v3-e2e",
            format_output_agent=True,
        )
        self.assertEqual(
            f"<answer>{rows[0]['canonical_answer']}</answer>",
            result.final_answer,
        )
        self.assertEqual([(QUESTION, 3)], index.search_calls)
        self.assertEqual(memory_ids, index.read_calls)

        calls = {call.request.agent.id: call for call in result.calls}
        retriever = calls["retriever"]
        reasoner = calls["reasoner"]
        verifier = calls["verifier"]
        formatter = calls["formatter"]
        self.assertEqual("evidence_retriever", retriever.request.agent.role_family)
        projected_retriever_artifact = json.loads(retriever.response.text)
        self.assertEqual(QUESTION, projected_retriever_artifact["question_scope"])
        self.assertEqual(QUESTION, projected_retriever_artifact["retrieval_query"])
        self.assertEqual(3, projected_retriever_artifact["top_k"])
        self.assertEqual(
            memory_ids,
            [
                candidate["memory_id"]
                for candidate in projected_retriever_artifact["candidates"]
            ],
        )
        for rank, (row, candidate) in enumerate(
            zip(rows, projected_retriever_artifact["candidates"], strict=True),
            start=1,
        ):
            self.assertEqual(rank, candidate["rank"])
            self.assertAlmostEqual(1.0 - (rank / 100), candidate["similarity"])
            for field in (
                "memory_id",
                "source_train_task_id",
                "paraphrase_question",
                "paraphrase_answer_statement",
                "canonical_answer",
            ):
                self.assertEqual(row[field], candidate[field])
        completion_schema = json.loads(
            react_gateway.requests[-1].model.metadata["response_json_schema"]
        )
        completion_value_schema = completion_schema["properties"]["arguments"][
            "properties"
        ]["value"]
        self.assertEqual(["memory_ids"], completion_value_schema["required"])
        self.assertEqual(
            {"memory_ids"}, set(completion_value_schema["properties"])
        )
        self.assertEqual(
            (TRIVIAQA_QA_MEMORY_TOOL_ID,),
            retriever.request.agent.allowed_tools,
        )
        self.assertEqual("reasoning", reasoner.request.agent.execution_mode.value)
        self.assertEqual((), reasoner.request.agent.allowed_tools)
        self.assertEqual("retriever", reasoner.request.upstream[0].source_agent_id)
        self.assertTrue(
            graph.relation_bits("retriever", "reasoner").source_to_target
        )
        self.assertEqual(4, len(reasoner.request.upstream[0].tool_receipts))
        self.assertEqual(
            projected_retriever_artifact,
            json.loads(reasoner.request.upstream[0].artifact),
        )
        self.assertIsNotNone(reasoner.request.upstream[0].artifact_version)
        self.assertEqual("reasoner", verifier.request.upstream[0].source_agent_id)
        self.assertEqual("verifier", formatter.request.upstream[0].source_agent_id)
        self.assertEqual(4, len(formatter.request.upstream[0].tool_receipts))
        self.assertIsNotNone(formatter.request.upstream[0].artifact_version)

        trajectory = {
            "turns": [
                {
                    "round_index": 0,
                    "prompt": director_prompt,
                    "policy_response": '{"action":"add_subgraph"}',
                    "action": {"action": "add_subgraph"},
                    "canvas_feedback": "accepted; typed execution receipt available",
                    "graph_snapshot": graph.to_dict(),
                    "executions": [
                        execution_record_from_call(call).to_dict()
                        for call in result.calls
                    ],
                    "runtime_summary": {
                        "output_agent_id": result.output_agent_id,
                        "final_answer": result.final_answer,
                    },
                }
            ]
        }
        control = analysis._trajectory_control_plane(  # noqa: SLF001
            "triviaqa:v3:e2e", trajectory
        )
        self.assertEqual(0, control["director_tool_calls"])
        self.assertEqual([], control["director_allowed_tools"])
        self.assertEqual([], control["director_execution_profile_violations"])
        self.assertGreater(control["retrieval_tool_call_count"], 0)
        self.assertEqual([], control["worker_ownership_violations"])
        self.assertEqual(
            [], control["reasoner_qamemory_tool_assignment_violations"]
        )
        self.assertEqual(
            1, control["native_artifact_receipt_projection_count"]
        )
        self.assertEqual(
            0,
            control["native_artifact_receipt_projection_violation_count"],
        )
        self.assertTrue(control["retrieval_artifact_routed_via_relation"])
        self.assertTrue(control["output_inbox_receipt_lineage"])

        # Reproduce tc_1 after one non-destructive Reasoner repair: execution
        # succeeded, the sole routed memory is semantically incompatible, and
        # the preserved Reasoner artifact therefore has no candidate.  The
        # next Canvas boundary must expose one bounded isolated Retriever
        # augmentation instead of canvas_action_domain_exhausted.
        env._progressive_outputs = dict(result.outputs)
        env._progressive_output_metadata = {
            agent_id: dict(metadata)
            for agent_id, metadata in result.output_metadata.items()
        }
        null_candidate = json.loads(env._progressive_outputs["reasoner"])
        null_candidate["candidate_answer"] = None
        env._progressive_outputs["reasoner"] = json.dumps(
            null_candidate,
            ensure_ascii=False,
        )
        env._repair_exhausted_agent_ids.add("reasoner")
        self.assertTrue(
            env._reasoner_failure_requires_evidence_augmentation("reasoner")
        )
        self.assertEqual(
            ("add_subgraph",), env.model_admissible_action_types()
        )
        self.assertEqual(
            ["evidence_retriever"],
            env.model_admissible_action_targets()["add_subgraph"][
                "admitted_new_role_families"
            ],
        )

        # Once a bounded worker schedule emits an established terminal
        # retrieval diagnosis, no additional Retriever generation is legal.
        # The outer evaluator may score the preserved empty answer as a valid
        # failure result; the Canvas must not turn it into an unbounded ADD
        # loop or permit a guessed semantic answer.
        for failure_code in (
            "knowledge_base_coverage_failure",
            "retrieval_strategy_failure",
        ):
            with self.subTest(terminal_retrieval_failure=failure_code):
                failure = AgentFailureRecord(
                    request_id=f"retriever-{failure_code}",
                    agent_id="retriever",
                    phase=ExecutionPhase.SINGLE,
                    graph_revision=graph.revision,
                    error_type="ReactExecutionError",
                    message="bounded retrieval schedule exhausted",
                    metadata={
                        "react_trace": [
                            {
                                "turn": 9,
                                "observation_status": "budget_exhausted",
                                "public_error_code": failure_code,
                                "terminal_failure_diagnosis": {
                                    "public_error_code": failure_code,
                                    "tool_plan_exhausted": True,
                                    "bounded_schedule_exhausted": True,
                                },
                            }
                        ],
                        "tool_receipts": list(
                            result.output_metadata["retriever"]["tool_receipts"]
                        ),
                    },
                )
                env._failed_agent_ids.add("retriever")
                env._repair_exhausted_agent_ids.add("retriever")
                env._latest_failure_record_by_agent["retriever"] = failure
                self.assertEqual(
                    failure_code,
                    env._typed_retrieval_failure_category(failure),
                )
                self.assertEqual((), env.model_admissible_action_types())
                self.assertNotIn(
                    "add_subgraph", env.model_admissible_action_targets()
                )


if __name__ == "__main__":
    unittest.main()
