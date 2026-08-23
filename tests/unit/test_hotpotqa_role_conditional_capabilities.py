"""Local contracts for the optional HotpotQA semantic capabilities protocol.

The fixtures are synthetic.  They exercise only the Canvas, action-domain,
prompt-observation, and terminal-admission interfaces and never invoke a model
or an external Tool.
"""

from __future__ import annotations

import json
import unittest

from src.interactive.agent_graph import AgentGraph, AgentNode, AgentRelation
from src.interactive.agent_runtime import (
    AgentFailureRecord,
    AgentRequest,
    AgentRuntime,
    AgentRuntimeResult,
    ExecutionPhase,
    ReasoningExecutionAdapter,
)
from src.interactive.agent_workflow_env import (
    AgentWorkflowEnv,
    AgentWorkflowStateError,
    _HOTPOTQA_ROLE_CONDITIONAL_FORMAT_CONTRACT,
)
from src.interactive.director import (
    DIRECTOR_PROMPT_VERSION,
    AgentGraphOrchestrator,
    director_live_action_parameter_json_schema_text,
)
from src.interactive.model_registry import (
    ModelRegistry,
    ModelSpec,
    ProviderSpec,
)
from src.interactive.openai_gateway import build_agent_messages
from src.interactive.qa_tool_adapter import (
    QARetrievalReactExecutionAdapter,
    QA_RETRIEVAL_TOOL_ID,
    build_qa_tool_registry,
)


SEMANTIC_PROTOCOL = "hotpotqa_role_conditional_capabilities_v1"
RECOVERY_POLICY = "preserve_diagnose_repair_augment"
SYNTHETIC_QUESTION = "Which target is reached through Bridge Beta?"
SYNTHETIC_CANDIDATE = "Target Gamma"


class _NoModelGateway:
    async def generate(self, request: AgentRequest) -> str:
        raise AssertionError(
            f"local contract must not invoke Agent {request.agent.id!r}"
        )


class _NoDirectorClient:
    async def propose(self, prompt: str, **kwargs: object) -> object:
        del prompt, kwargs
        raise AssertionError("local contract must not invoke the Director")


class _NoopRetrievalIndex:
    manifest = type(
        "Manifest",
        (),
        {
            "corpus_name": "synthetic-public-corpus",
            "corpus_version": "synthetic-v1",
            "index_id": "synthetic-index-v1",
            "format": "synthetic-retrieval-index@1",
            "retrieval_backend": "test",
        },
    )()

    def search(self, query: str, *, limit: int) -> tuple[object, ...]:
        del query, limit
        return ()

    def read(self, passage_id: str) -> object:
        raise AssertionError(f"local contract must not read {passage_id!r}")


def _registry() -> ModelRegistry:
    return ModelRegistry(
        [ProviderSpec("local", kind="test")],
        [
            ModelSpec("model-a", "local"),
            ModelSpec("model-b", "local"),
            ModelSpec("model-c", "local"),
        ],
    )


def _runtime(registry: ModelRegistry) -> AgentRuntime:
    gateway = _NoModelGateway()
    return AgentRuntime(
        registry,
        gateway,
        execution_adapters={"react": ReasoningExecutionAdapter(gateway)},
        tool_registry=build_qa_tool_registry(_NoopRetrievalIndex()),
        dataset_id="hotpotqa",
        semantic_protocol=SEMANTIC_PROTOCOL,
    )


def _env(
    registry: ModelRegistry,
    *,
    graph: AgentGraph | None = None,
    max_agents: int | None = None,
) -> AgentWorkflowEnv:
    return AgentWorkflowEnv(
        registry,
        runtime=_runtime(registry),
        graph=graph,
        max_agents=max_agents,
        problem=SYNTHETIC_QUESTION,
        require_exact_answer_tag=True,
        require_format_agent=False,
        semantic_protocol=SEMANTIC_PROTOCOL,
        recovery_policy=RECOVERY_POLICY,
        required_evidence_tool_id=QA_RETRIEVAL_TOOL_ID,
    )


def _read_receipt() -> dict[str, object]:
    return {
        "tool_id": QA_RETRIEVAL_TOOL_ID,
        "tool_version": "synthetic-v1",
        "request": {
            "action": "read",
            "arguments": {"passage_id": "synthetic-passage"},
        },
        "result": {
            "value": {
                "operation": "read",
                "passage": {
                    "id": "synthetic-passage",
                    "text": (
                        "Source Alpha links to Bridge Beta. "
                        f"Bridge Beta identifies {SYNTHETIC_CANDIDATE}."
                    ),
                },
            },
            "completed": True,
        },
        "error_type": None,
    }


def _retrieval_artifact() -> str:
    return json.dumps(
        [
            {
                "passage_id": "synthetic-passage",
                "evidence_span": (
                    "Source Alpha links to Bridge Beta. "
                    f"Bridge Beta identifies {SYNTHETIC_CANDIDATE}."
                ),
            }
        ],
        sort_keys=True,
    )


def _reasoner_artifact(
    candidate: str = SYNTHETIC_CANDIDATE,
    *,
    question: str = SYNTHETIC_QUESTION,
    answer_type: str = "entity",
) -> str:
    return json.dumps(
        {
            "question_scope": question,
            "answer_slot": {
                "answer_type": answer_type,
                "answer_cardinality": "single",
                "qualifiers": ["through Bridge Beta"],
                "proposition_index": 1,
                "answer_field": "object_or_attribute_value",
            },
            "evidence_propositions": [
                {
                    "subject": "Source Alpha",
                    "relation": "links to",
                    "object_or_attribute_value": "Bridge Beta",
                    "qualifiers": [],
                    "evidence_span": "Source Alpha links to Bridge Beta.",
                },
                {
                    "subject": "Bridge Beta",
                    "relation": "identifies",
                    "object_or_attribute_value": candidate,
                    "qualifiers": [],
                    "evidence_span": f"Bridge Beta identifies {candidate}.",
                },
            ],
            "multi_hop_chain": [
                "Source Alpha links to Bridge Beta",
                f"Bridge Beta identifies {candidate}",
            ],
            "candidate_answer": candidate,
            "evidence": [
                "Source Alpha links to Bridge Beta.",
                f"Bridge Beta identifies {candidate}.",
            ],
        },
        sort_keys=True,
    )


def _verifier_artifact(candidate: str = SYNTHETIC_CANDIDATE) -> str:
    return json.dumps(
        {
            "candidate_answer": candidate,
            "evidence_supported": True,
            "entity_attribute_binding_correct": True,
            "alias_binding_correct": True,
            "answer_type_cardinality_correct": True,
            "multi_hop_complete": True,
            "minimal_answer_surface": True,
            "scope_preserved": True,
            "verification_status": "supported",
        },
        sort_keys=True,
    )


def _execution(
    graph: AgentGraph,
    *,
    outputs: dict[str, str],
    final_answer: str,
    receipt_agent_ids: tuple[str, ...] = (),
) -> AgentRuntimeResult:
    return AgentRuntimeResult(
        run_id="synthetic-run",
        graph_revision=graph.revision,
        output_agent_id=graph.output_agent_id,
        final_answer=final_answer,
        outputs=outputs,
        calls=(),
        block_completion_order=(),
        output_metadata={
            agent_id: {"tool_receipts": [_read_receipt()]}
            for agent_id in receipt_agent_ids
        },
    )


def _evidence_agent(agent_id: str = "retriever") -> AgentNode:
    return AgentNode(
        agent_id,
        "model-a",
        "retrieve explicit evidence for the original question",
        role_family="evidence_retriever",
        allowed_tools=(QA_RETRIEVAL_TOOL_ID,),
        execution_mode="react",
        artifact_type="retrieval_evidence",
    )


def _output_agent(agent_id: str = "output") -> AgentNode:
    return AgentNode(
        agent_id,
        "model-b",
        "produce the exact terminal answer from routed evidence",
        role_family="output",
        execution_mode="reasoning",
        artifact_type="answer_wrapper",
    )


class RoleConditionalSearchSpaceTests(unittest.TestCase):
    def test_semantic_roles_are_optional_and_react_is_not_a_role(self) -> None:
        registry = _registry()
        env = _env(registry)

        self.assertEqual((), env._missing_semantic_role_families())
        add_domain = env.model_admissible_action_targets()["add_subgraph"]
        admitted_roles = add_domain["admitted_new_role_families"]
        self.assertIn("reasoner", admitted_roles)
        self.assertIn("verifier", admitted_roles)
        self.assertIn("format", admitted_roles)
        self.assertIn("output", admitted_roles)
        self.assertNotIn("react", admitted_roles)
        self.assertIs(add_domain["distinct_new_role_families"], False)
        self.assertIs(add_domain["defer_output_assignment"], False)

        invalid = AgentGraph(
            [
                AgentNode(
                    "invalid",
                    "model-a",
                    "retrieve evidence",
                    role_family="react",
                    allowed_tools=(QA_RETRIEVAL_TOOL_ID,),
                    execution_mode="react",
                )
            ]
        )
        issue = env._semantic_edit_issue_for(invalid)
        self.assertIsNotNone(issue)
        self.assertIn("ReAct is execution_mode='react'", issue or "")

    def test_reasoner_supports_reasoning_or_react_execution(self) -> None:
        registry = _registry()
        env = _env(registry)
        reasoner_domain = env.model_admissible_action_targets()[
            "add_subgraph"
        ]["role_constraints"]["reasoner"]

        self.assertEqual(
            ["reasoning", "react"],
            reasoner_domain["execution_modes"],
        )
        self.assertEqual(
            [[], [QA_RETRIEVAL_TOOL_ID]],
            reasoner_domain["allowed_tools"],
        )

        routed_reasoner = AgentGraph(
            [
                AgentNode(
                    "reasoner",
                    "model-a",
                    "align routed evidence to the requested answer slot",
                    role_family="reasoner",
                    execution_mode="reasoning",
                )
            ]
        )
        react_reasoner = AgentGraph(
            [
                AgentNode(
                    "reasoner",
                    "model-a",
                    "retrieve evidence and align it to the requested answer slot",
                    role_family="reasoner",
                    allowed_tools=(QA_RETRIEVAL_TOOL_ID,),
                    execution_mode="react",
                )
            ]
        )
        self.assertIsNone(env._semantic_edit_issue_for(routed_reasoner))
        self.assertIsNone(env._semantic_edit_issue_for(react_reasoner))

    def test_react_reasoner_reuses_hotpot_structured_completion_and_two_reads(
        self,
    ) -> None:
        adapter = QARetrievalReactExecutionAdapter(
            gateway=_NoModelGateway(),
            tool_registry=build_qa_tool_registry(_NoopRetrievalIndex()),
            max_turns=9,
            max_tool_calls=6,
            task_type="multi_hop_qa",
            completion_policy="required_evidence",
        )
        request = AgentRequest(
            request_id="hotpot:role-conditional-reasoner",
            run_id="hotpot",
            graph_revision=1,
            problem="Which entity has the larger value?",
            agent=AgentNode(
                "reasoner",
                "model-a",
                "retrieve evidence and align it to the requested answer slot",
                role_family="reasoner",
                allowed_tools=(QA_RETRIEVAL_TOOL_ID,),
                execution_mode="react",
            ),
            model=_registry().require_model("model-a"),
            provider=_registry().provider_for("model-a"),
            phase=ExecutionPhase.SINGLE,
            semantic_protocol=SEMANTIC_PROTOCOL,
        )
        schema = adapter._completion_arguments_schema(request)
        semantic_fields = schema["properties"]["value"]["properties"]
        self.assertEqual(
            {
                "question_scope",
                "answer_slot",
                "evidence_propositions",
                "multi_hop_chain",
                "candidate_answer",
                "evidence",
            },
            set(semantic_fields),
        )
        first_read = {
            "observation_status": "success",
            "result": {
                "operation": "read",
                "passage_id": "p1",
                "passage": {"text": "Entity A has value 10."},
            },
        }
        second_read = {
            "observation_status": "success",
            "result": {
                "operation": "read",
                "passage_id": "p2",
                "passage": {"text": "Entity B has value 12."},
            },
        }
        _, completion_after_one_read = adapter._state_conditioned_action_domain(
            request,
            [first_read],
        )
        _, completion_after_two_reads = adapter._state_conditioned_action_domain(
            request,
            [first_read, second_read],
        )
        self.assertIs(completion_after_one_read, False)
        self.assertIs(completion_after_two_reads, True)

    def test_retriever_exports_only_receipt_grounded_evidence_citations(
        self,
    ) -> None:
        adapter = QARetrievalReactExecutionAdapter(
            gateway=_NoModelGateway(),
            tool_registry=build_qa_tool_registry(_NoopRetrievalIndex()),
            max_turns=9,
            max_tool_calls=6,
            task_type="multi_hop_qa",
            completion_policy="required_evidence",
        )
        request = AgentRequest(
            request_id="hotpot:role-conditional-retriever",
            run_id="hotpot",
            graph_revision=1,
            problem=SYNTHETIC_QUESTION,
            agent=_evidence_agent(),
            model=_registry().require_model("model-a"),
            provider=_registry().provider_for("model-a"),
            phase=ExecutionPhase.SINGLE,
            semantic_protocol=SEMANTIC_PROTOCOL,
        )
        schema = adapter._completion_arguments_schema(request)
        citation_schema = schema["properties"]["value"]
        self.assertEqual("array", citation_schema["type"])
        self.assertEqual(1, citation_schema["minItems"])
        self.assertEqual(
            {"passage_id", "evidence_span"},
            set(citation_schema["items"]["properties"]),
        )
        serialized_schema = json.dumps(schema, sort_keys=True)
        for forbidden in (
            "entity_identity",
            "target_relation",
            "answer_type_constraint",
            "evidence_proposition",
            "answer_slot",
            "candidate_answer",
            "final_answer",
        ):
            self.assertNotIn(forbidden, serialized_schema)

        evidence_span = (
            "Arthur's Magazine (1844–1846) was an American literary "
            "periodical published in Philadelphia."
        )
        receipt = {
            "tool_id": QA_RETRIEVAL_TOOL_ID,
            "tool_version": "synthetic-v1",
            "request": {
                "action": "read",
                "arguments": {"passage_id": "arthurs-magazine"},
            },
            "result": {
                "value": {
                    "operation": "read",
                    "passage_id": "arthurs-magazine",
                    "passage": {
                        "passage_id": "arthurs-magazine",
                        "text": evidence_span,
                    },
                },
                "completed": True,
            },
            "error_type": None,
        }
        artifact = json.dumps(
            [
                {
                    "passage_id": "arthurs-magazine",
                    "evidence_span": evidence_span,
                }
            ]
        )
        self.assertIsNone(
            adapter._role_conditional_evidence_completion_issue(
                artifact=artifact,
                tool_receipts=(receipt,),
            )
        )
        bad_artifact = json.dumps(
            [
                {
                    "passage_id": "arthurs-magazine",
                    "evidence_span": "Arthur's Magazine started in 1844.",
                }
            ]
        )
        self.assertIn(
            "no typography-canonical lexical match",
            adapter._role_conditional_evidence_completion_issue(
                artifact=bad_artifact,
                tool_receipts=(receipt,),
            )
            or "",
        )

    def test_non_formatter_is_a_legal_output(self) -> None:
        registry = _registry()
        graph = AgentGraph(
            [_evidence_agent(), _output_agent()],
            [AgentRelation("retriever", "output", True, False)],
        )
        env = _env(registry, graph=graph)

        self.assertIn("output", env._model_admissible_output_agent_ids())
        candidate = graph.fork()
        candidate.set_output("output")
        self.assertIsNone(env._semantic_edit_issue_for(candidate))
        self.assertIsNone(env._format_agent_issue_for(candidate))
        self.assertEqual((), env._required_semantic_edges())

        add_domain = env.model_admissible_action_targets()["add_subgraph"]
        self.assertEqual(
            ["format", "output"],
            add_domain["output_role_families"],
        )

    def test_add_subgraph_can_select_a_terminal_output_after_a_reasoner(self) -> None:
        registry = _registry()
        graph = AgentGraph(
            [
                AgentNode(
                    "reasoner",
                    "model-a",
                    "retrieve evidence and align it to the requested answer slot",
                    role_family="reasoner",
                    allowed_tools=(QA_RETRIEVAL_TOOL_ID,),
                    execution_mode="react",
                )
            ],
        )
        env = _env(registry, graph=graph)
        domains = env.model_admissible_action_targets()
        declaration = {
            "agent_id": "node_1",
            "model_id": "model-b",
            "contract": "produce the exact terminal answer from routed evidence",
            "role_family": "output",
            "allowed_tools": [],
            "execution_mode": "reasoning",
        }
        schema = json.loads(
            director_live_action_parameter_json_schema_text(
                "add_subgraph",
                domains,
                add_agents=(declaration,),
            )
        )
        output_schema = schema["properties"]["output_agent_id"]
        self.assertNotIn("output_agent_id", schema["required"])
        output_enums = {
            item
            for branch in output_schema["anyOf"]
            for item in branch.get("enum", ())
        }
        self.assertIn("node_1", output_enums)

        candidate = graph.fork()
        candidate.add_agent(
            AgentNode(
                "node_1",
                "model-b",
                declaration["contract"],
                role_family="output",
                execution_mode="reasoning",
            )
        )
        candidate.set_relation("reasoner", "node_1", True, False)
        candidate.set_output("node_1")
        self.assertIsNone(env._output_sink_issue_for(candidate))
        self.assertIsNone(env._semantic_edit_issue_for(candidate))

    def test_generic_output_without_reasoner_can_receive_evidence_ingress(
        self,
    ) -> None:
        registry = _registry()
        graph = AgentGraph([_output_agent()], output_agent_id="output")
        env = _env(registry, graph=graph)
        outputs = {"output": f"<answer>{SYNTHETIC_CANDIDATE}</answer>"}
        env._progressive_execution = _execution(
            graph,
            outputs=outputs,
            final_answer=outputs["output"],
        )
        env._progressive_execution_revision = graph.revision
        env._progressive_outputs = dict(outputs)

        self.assertEqual(
            ("output",),
            env._role_conditional_evidence_ingress_consumer_ids(),
        )
        domain = env.model_admissible_action_targets()["add_subgraph"]
        self.assertEqual(
            ["output"],
            domain["required_ingress_consumer_agent_ids"],
        )
        self.assertEqual(
            ["evidence_retriever"],
            domain["admitted_new_role_families"],
        )
        self.assertIs(domain["explicit_output_assignment_required"], False)

    def test_open_add_preserves_an_existing_output_without_reassignment(self) -> None:
        registry = _registry()
        graph = AgentGraph(
            [
                _evidence_agent(),
                AgentNode(
                    "reasoner",
                    "model-b",
                    "align routed evidence to the requested answer slot",
                    role_family="reasoner",
                    execution_mode="reasoning",
                ),
                _output_agent(),
            ],
            [
                AgentRelation("retriever", "reasoner", True, False),
                AgentRelation("reasoner", "output", True, False),
            ],
            output_agent_id="output",
        )
        env = _env(registry, graph=graph)
        outputs = {
            "retriever": _retrieval_artifact(),
            "reasoner": "malformed semantic artifact",
            "output": f"<answer>{SYNTHETIC_CANDIDATE}</answer>",
        }
        env._progressive_execution = _execution(
            graph,
            outputs=outputs,
            final_answer=outputs["output"],
            receipt_agent_ids=("retriever",),
        )
        env._progressive_execution_revision = graph.revision
        env._progressive_outputs = dict(outputs)

        domain = env.model_admissible_action_targets()["add_subgraph"]
        self.assertEqual("output", domain["current_output_agent_id"])
        self.assertIs(domain["explicit_output_assignment_required"], False)
        declaration = {
            "agent_id": "node_1",
            "model_id": "model-c",
            "contract": "repair the malformed semantic artifact",
            "role_family": "repair",
            "allowed_tools": [],
            "execution_mode": "reasoning",
        }
        schema = json.loads(
            director_live_action_parameter_json_schema_text(
                "add_subgraph",
                {"add_subgraph": domain},
                add_agents=(declaration,),
            )
        )
        self.assertNotIn("output_agent_id", schema["required"])
        self.assertEqual(
            {"type": "null"},
            schema["properties"]["output_agent_id"],
        )

    def test_each_reasoner_requires_evidence_on_its_own_routed_path(self) -> None:
        registry = _registry()
        graph = AgentGraph(
            [
                _evidence_agent(),
                AgentNode(
                    "reasoner_a",
                    "model-a",
                    "align routed evidence to the requested answer slot",
                    role_family="reasoner",
                    execution_mode="reasoning",
                    artifact_type="semantic_candidate",
                ),
                AgentNode(
                    "reasoner_b",
                    "model-b",
                    "independently align evidence to the requested answer slot",
                    role_family="reasoner",
                    execution_mode="reasoning",
                    artifact_type="semantic_candidate",
                ),
                _output_agent(),
            ],
            [
                AgentRelation("retriever", "reasoner_a", True, False),
                AgentRelation("reasoner_a", "output", True, False),
                AgentRelation("reasoner_b", "output", True, False),
            ],
            output_agent_id="output",
        )
        env = _env(registry, graph=graph)
        outputs = {
            "retriever": "retrieved evidence",
            "reasoner_a": _reasoner_artifact(),
            "reasoner_b": _reasoner_artifact(),
            "output": f"<answer>{SYNTHETIC_CANDIDATE}</answer>",
        }
        env._progressive_execution = _execution(
            graph,
            outputs=outputs,
            final_answer=outputs["output"],
            receipt_agent_ids=("retriever",),
        )
        env._progressive_execution_revision = graph.revision
        env._progressive_outputs = dict(outputs)

        self.assertEqual(
            ("reasoner_b",),
            env._role_conditional_evidence_ingress_consumer_ids(),
        )
        self.assertEqual(("set_relation",), env.model_admissible_action_types())
        candidates = env._model_admissible_relation_candidates()
        self.assertEqual(
            [
                {
                    "source_id": "retriever",
                    "target_id": "reasoner_b",
                    "source_to_target": True,
                    "target_to_source": False,
                }
            ],
            candidates,
        )
        candidate = graph.fork()
        candidate.set_relation("retriever", "reasoner_b", True, False)
        self.assertTrue(
            env._role_conditional_existing_evidence_ingress_candidate(
                candidate
            )
        )
        self.assertIsNone(env._preserved_input_change_issue_for(candidate))

        finished_env = _env(registry, graph=candidate)
        finished_execution = _execution(
            candidate,
            outputs=outputs,
            final_answer=outputs["output"],
            receipt_agent_ids=("retriever",),
        )
        finished_env._progressive_execution = finished_execution
        finished_env._progressive_execution_revision = candidate.revision
        finished_env._progressive_outputs = dict(outputs)
        self.assertIsNone(
            finished_env._semantic_protocol_issue(finished_execution)
        )
        self.assertIs(
            finished_env.finish_admissibility()["admissible"],
            True,
        )

    def test_receipt_bearing_reasoner_repairs_parallel_reasoner_at_capacity(
        self,
    ) -> None:
        registry = _registry()
        graph = AgentGraph(
            [
                AgentNode(
                    "reasoner_a",
                    "model-a",
                    "retrieve evidence and determine one semantic candidate",
                    role_family="reasoner",
                    allowed_tools=(QA_RETRIEVAL_TOOL_ID,),
                    execution_mode="react",
                    artifact_type="semantic_candidate",
                ),
                AgentNode(
                    "reasoner_b",
                    "model-b",
                    "independently align evidence to the requested answer slot",
                    role_family="reasoner",
                    execution_mode="reasoning",
                    artifact_type="semantic_candidate",
                ),
                _output_agent(),
            ],
            [
                AgentRelation("reasoner_a", "output", True, False),
                AgentRelation("reasoner_b", "output", True, False),
            ],
            output_agent_id="output",
        )
        env = _env(registry, graph=graph, max_agents=3)
        outputs = {
            "reasoner_a": _reasoner_artifact(),
            "reasoner_b": _reasoner_artifact(),
            "output": f"<answer>{SYNTHETIC_CANDIDATE}</answer>",
        }
        env._progressive_execution = _execution(
            graph,
            outputs=outputs,
            final_answer=outputs["output"],
            receipt_agent_ids=("reasoner_a",),
        )
        env._progressive_execution_revision = graph.revision
        env._progressive_outputs = dict(outputs)

        self.assertEqual(
            ("reasoner_b",),
            env._role_conditional_evidence_ingress_consumer_ids(),
        )
        self.assertEqual(("set_relation",), env.model_admissible_action_types())
        self.assertEqual(
            [
                {
                    "source_id": "reasoner_a",
                    "target_id": "reasoner_b",
                    "source_to_target": True,
                    "target_to_source": False,
                }
            ],
            env._model_admissible_relation_candidates(),
        )

    def test_existing_evidence_can_complete_a_bounded_reciprocal_block(
        self,
    ) -> None:
        registry = _registry()
        graph = AgentGraph(
            [
                AgentNode(
                    "reasoner",
                    "model-a",
                    "align evidence to the requested answer slot",
                    role_family="reasoner",
                    execution_mode="reasoning",
                    artifact_type="semantic_candidate",
                ),
                _evidence_agent(),
                _output_agent(),
            ],
            [
                AgentRelation("reasoner", "retriever", True, False),
                AgentRelation("retriever", "output", True, False),
            ],
            output_agent_id="output",
        )
        env = _env(registry, graph=graph, max_agents=3)
        outputs = {
            "reasoner": _reasoner_artifact(),
            "retriever": "retrieved evidence",
            "output": f"<answer>{SYNTHETIC_CANDIDATE}</answer>",
        }
        env._progressive_execution = _execution(
            graph,
            outputs=outputs,
            final_answer=outputs["output"],
            receipt_agent_ids=("retriever",),
        )
        env._progressive_execution_revision = graph.revision
        env._progressive_outputs = dict(outputs)

        self.assertEqual(("reasoner",), env._role_conditional_evidence_ingress_consumer_ids())
        self.assertEqual(("set_relation",), env.model_admissible_action_types())
        candidates = env._model_admissible_relation_candidates()
        self.assertEqual(1, len(candidates))
        self.assertIs(candidates[0]["source_to_target"], True)
        self.assertIs(candidates[0]["target_to_source"], True)
        candidate = graph.fork()
        candidate.set_relation("reasoner", "retriever", True, True)
        self.assertTrue(
            env._role_conditional_existing_evidence_ingress_candidate(
                candidate
            )
        )
        self.assertIsNone(env._preserved_input_change_issue_for(candidate))

    def test_missing_evidence_projects_one_retriever_ingress_augmentation(
        self,
    ) -> None:
        registry = _registry()
        graph = AgentGraph(
            [
                AgentNode(
                    "reasoner",
                    "model-a",
                    "align routed evidence to the requested answer slot",
                    role_family="reasoner",
                    execution_mode="reasoning",
                    artifact_type="semantic_candidate",
                ),
                AgentNode(
                    "verifier",
                    "model-b",
                    "verify the routed semantic candidate",
                    role_family="verifier",
                    execution_mode="reasoning",
                    artifact_type="verification_report",
                ),
                _output_agent(),
            ],
            [
                AgentRelation("reasoner", "verifier", True, False),
                AgentRelation("verifier", "output", True, False),
            ],
            output_agent_id="output",
        )
        env = _env(registry, graph=graph)
        outputs = {
            "reasoner": _reasoner_artifact(),
            "verifier": _verifier_artifact(),
            "output": f"<answer>{SYNTHETIC_CANDIDATE}</answer>",
        }
        env._progressive_execution = _execution(
            graph,
            outputs=outputs,
            final_answer=outputs["output"],
        )
        env._progressive_execution_revision = graph.revision
        env._progressive_outputs = dict(outputs)

        self.assertEqual(("add_subgraph",), env.model_admissible_action_types())
        domain = env.model_admissible_action_targets()["add_subgraph"]
        self.assertEqual(1, domain["max_new_agents"])
        self.assertEqual(
            ["evidence_retriever"],
            domain["admitted_new_role_families"],
        )
        self.assertEqual(
            ["reasoner"],
            domain["required_ingress_consumer_agent_ids"],
        )
        self.assertEqual(1, domain["exact_relation_count"])
        self.assertIs(domain["explicit_output_assignment_required"], False)
        declaration = {
            "agent_id": "node_1",
            "model_id": "model-c",
            "contract": "retrieve explicit evidence for the original question",
            "role_family": "evidence_retriever",
            "allowed_tools": [QA_RETRIEVAL_TOOL_ID],
            "execution_mode": "react",
        }
        schema = json.loads(
            director_live_action_parameter_json_schema_text(
                "add_subgraph",
                {"add_subgraph": domain},
                add_agents=(declaration,),
            )
        )
        self.assertEqual({"type": "null"}, schema["properties"]["output_agent_id"])
        self.assertEqual(1, schema["properties"]["relations"]["minItems"])
        self.assertEqual(1, schema["properties"]["relations"]["maxItems"])
        for branch in schema["properties"]["relations"]["items"]["anyOf"]:
            relation = {
                key: value["const"]
                for key, value in branch["properties"].items()
            }
            self.assertEqual(
                {
                    "source_id": "node_1",
                    "target_id": "reasoner",
                    "source_to_target": True,
                    "target_to_source": False,
                },
                relation,
            )

        candidate = graph.fork()
        candidate.add_agent(
            AgentNode(
                "node_1",
                "model-c",
                declaration["contract"],
                role_family="evidence_retriever",
                allowed_tools=(QA_RETRIEVAL_TOOL_ID,),
                execution_mode="react",
            )
        )
        candidate.set_relation("node_1", "reasoner", True, False)
        self.assertTrue(env._role_conditional_evidence_ingress_candidate(candidate))
        self.assertIsNone(env._preserved_input_change_issue_for(candidate))
        valid_payload = {
            "action": "add_subgraph",
            "agents": [declaration],
            "relations": [
                {
                    "source_id": "node_1",
                    "target_id": "reasoner",
                    "source_to_target": True,
                    "target_to_source": False,
                }
            ],
            "output_agent_id": None,
        }
        valid_action = env.parser.parse(json.dumps(valid_payload))
        self.assertIsNone(
            env._role_conditional_evidence_ingress_admission_issue(
                valid_action
            )
        )
        for invalid_relations in (
            [],
            [
                *valid_payload["relations"],
                {
                    "source_id": "node_1",
                    "target_id": "verifier",
                    "source_to_target": True,
                    "target_to_source": False,
                },
            ],
        ):
            invalid_payload = {
                **valid_payload,
                "relations": invalid_relations,
            }
            invalid_action = env.parser.parse(json.dumps(invalid_payload))
            self.assertIsNotNone(
                env._role_conditional_evidence_ingress_admission_issue(
                    invalid_action
                )
            )

    def test_selected_verifier_requires_routed_input_only_at_output_boundary(
        self,
    ) -> None:
        registry = _registry()
        partial = AgentGraph(
            [
                AgentNode(
                    "verifier",
                    "model-a",
                    "verify an upstream semantic candidate",
                    role_family="verifier",
                    execution_mode="reasoning",
                )
            ]
        )
        env = _env(registry, graph=partial)
        self.assertIsNone(env._semantic_edit_issue_for(partial))

        terminal = partial.fork()
        terminal.set_output("verifier")
        issue = env._semantic_edit_issue_for(terminal)
        self.assertIsNotNone(issue)
        self.assertIn("terminal-compatible", issue or "")

    def test_deferred_consumer_projects_executable_ingress_repair(self) -> None:
        registry = _registry()
        graph = AgentGraph(
            [
                AgentNode(
                    "verifier",
                    "model-a",
                    "verify a routed semantic candidate",
                    role_family="verifier",
                    execution_mode="reasoning",
                )
            ]
        )
        env = _env(registry, graph=graph)
        env._progressive_execution = AgentRuntimeResult(
            run_id="deferred",
            graph_revision=graph.revision,
            output_agent_id=None,
            final_answer=None,
            outputs={},
            calls=(),
            block_completion_order=(),
            deferred_agent_ids=("verifier",),
        )
        env._progressive_execution_revision = graph.revision

        self.assertEqual(("add_subgraph",), env.model_admissible_action_types())
        add_domain = env.model_admissible_action_targets()["add_subgraph"]
        self.assertEqual(
            ["verifier"],
            add_domain["required_ingress_consumer_agent_ids"],
        )
        self.assertNotIn(
            "verifier", add_domain["admitted_new_role_families"]
        )
        self.assertNotIn("format", add_domain["admitted_new_role_families"])
        self.assertEqual(
            ["reasoner", "repair"],
            add_domain["admitted_new_role_families"],
        )
        declaration = {
            "agent_id": "node_1",
            "model_id": "model-b",
            "contract": "determine one semantic candidate for verification",
            "role_family": "repair",
            "allowed_tools": [],
            "execution_mode": "reasoning",
        }
        schema = json.loads(
            director_live_action_parameter_json_schema_text(
                "add_subgraph",
                {"add_subgraph": add_domain},
                add_agents=(declaration,),
            )
        )
        self.assertEqual(1, schema["properties"]["relations"]["minItems"])
        for branch in schema["properties"]["relations"]["items"]["anyOf"]:
            relation = {
                key: value["const"]
                for key, value in branch["properties"].items()
            }
            supplies_verifier = (
                relation["source_id"] == "node_1"
                and relation["target_id"] == "verifier"
                and relation["source_to_target"] is True
            ) or (
                relation["target_id"] == "node_1"
                and relation["source_id"] == "verifier"
                and relation["target_to_source"] is True
            )
            self.assertTrue(supplies_verifier)

    def test_materialized_semantic_agent_repairs_deferred_consumer_by_relation(
        self,
    ) -> None:
        registry = _registry()
        graph = AgentGraph(
            [
                AgentNode(
                    "semantic_producer",
                    "model-a",
                    "determine one semantic candidate",
                    role_family="repair",
                    execution_mode="reasoning",
                    artifact_type="semantic_candidate",
                ),
                AgentNode(
                    "verifier",
                    "model-b",
                    "verify a routed semantic candidate",
                    role_family="verifier",
                    execution_mode="reasoning",
                ),
            ]
        )
        env = _env(registry, graph=graph)
        env._progressive_execution = AgentRuntimeResult(
            run_id="deferred-existing",
            graph_revision=graph.revision,
            output_agent_id=None,
            final_answer=None,
            outputs={
                "semantic_producer": (
                    f"Candidate answer: {SYNTHETIC_CANDIDATE}"
                )
            },
            calls=(),
            block_completion_order=(),
            executed_agent_ids=("semantic_producer",),
            deferred_agent_ids=("verifier",),
        )
        env._progressive_execution_revision = graph.revision

        self.assertEqual(("set_relation",), env.model_admissible_action_types())
        candidates = env.model_admissible_action_targets()["set_relation"][
            "candidates"
        ]
        self.assertTrue(candidates)
        for relation in candidates:
            self.assertEqual("semantic_producer", relation["source_id"])
            self.assertEqual("verifier", relation["target_id"])
            self.assertIs(relation["source_to_target"], True)

    def test_raw_retrieval_does_not_satisfy_deferred_verifier_ingress(
        self,
    ) -> None:
        registry = _registry()
        graph = AgentGraph(
            [
                _evidence_agent(),
                AgentNode(
                    "verifier",
                    "model-b",
                    "verify one routed semantic candidate",
                    role_family="verifier",
                    execution_mode="reasoning",
                ),
            ]
        )
        env = _env(registry, graph=graph)
        env._progressive_execution = AgentRuntimeResult(
            run_id="deferred-raw-evidence",
            graph_revision=graph.revision,
            output_agent_id=None,
            final_answer=None,
            outputs={"retriever": "retrieved evidence"},
            calls=(),
            block_completion_order=(),
            executed_agent_ids=("retriever",),
            deferred_agent_ids=("verifier",),
        )
        env._progressive_execution_revision = graph.revision

        self.assertEqual(("add_subgraph",), env.model_admissible_action_types())
        domain = env.model_admissible_action_targets()["add_subgraph"]
        self.assertEqual(
            ["reasoner", "repair"],
            domain["admitted_new_role_families"],
        )

    def test_exhausted_auxiliary_replacement_is_an_isolated_add_boundary(
        self,
    ) -> None:
        registry = _registry()
        graph = AgentGraph(
            [
                _evidence_agent("reader"),
                AgentNode(
                    "reasoner",
                    "model-b",
                    "align routed evidence to the requested answer slot",
                    role_family="reasoner",
                    execution_mode="reasoning",
                ),
                _output_agent(),
            ],
            [
                AgentRelation("reader", "reasoner", True, False),
                AgentRelation("reasoner", "output", True, False),
            ],
            output_agent_id="output",
        )
        env = _env(registry, graph=graph)
        env._failed_agent_ids.add("reader")
        env._repair_exhausted_agent_ids.add("reader")
        env._react_exhausted_agent_ids.add("reader")
        env._latest_failure_record_by_agent["reader"] = AgentFailureRecord(
            request_id="reader-exhausted",
            agent_id="reader",
            phase=ExecutionPhase.SINGLE,
            graph_revision=graph.revision,
            error_type="ReactExecutionError",
            message="react agent 'reader' exhausted 9 turns",
        )

        self.assertEqual(("add_subgraph",), env.model_admissible_action_types())
        domain = env.model_admissible_action_targets()["add_subgraph"]
        self.assertEqual(1, domain["max_new_agents"])
        self.assertEqual(["evidence_retriever"], domain["admitted_new_role_families"])
        self.assertEqual([], domain["relations"])
        self.assertIsNone(domain["output_agent_id"])
        self.assertIs(domain["explicit_output_assignment_required"], False)
        declaration = {
            "agent_id": "node_1",
            "model_id": "model-c",
            "contract": "continue receipt-grounded evidence retrieval",
            "role_family": "evidence_retriever",
            "allowed_tools": [QA_RETRIEVAL_TOOL_ID],
            "execution_mode": "react",
            "artifact_type": "retrieval_evidence",
        }
        schema = json.loads(
            director_live_action_parameter_json_schema_text(
                "add_subgraph",
                {"add_subgraph": domain},
                add_agents=(declaration,),
            )
        )
        self.assertEqual({"type": "null"}, schema["properties"]["output_agent_id"])
        self.assertEqual({"type": "array", "maxItems": 0}, schema["properties"]["relations"])
        contradictory_domain = {
            **domain,
            "explicit_output_assignment_required": True,
        }
        with self.assertRaisesRegex(
            ValueError,
            "isolated replacement boundary cannot require",
        ):
            director_live_action_parameter_json_schema_text(
                "add_subgraph",
                {"add_subgraph": contradictory_domain},
                add_agents=(declaration,),
            )


class RoleConditionalTerminalTests(unittest.TestCase):
    def test_explicit_question_head_is_a_valid_answer_type_subtype(self) -> None:
        registry = _registry()
        question = "The hotel company has its head office in what city?"
        candidate = SYNTHETIC_CANDIDATE
        graph = AgentGraph(
            [
                _evidence_agent(),
                AgentNode(
                    "reasoner",
                    "model-b",
                    "align routed evidence to the requested answer slot",
                    role_family="reasoner",
                    execution_mode="reasoning",
                ),
                _output_agent(),
            ],
            [
                AgentRelation("retriever", "reasoner", True, False),
                AgentRelation("reasoner", "output", True, False),
            ],
            output_agent_id="output",
        )
        env = AgentWorkflowEnv(
            registry,
            runtime=_runtime(registry),
            graph=graph,
            problem=question,
            require_exact_answer_tag=True,
            require_format_agent=False,
            semantic_protocol=SEMANTIC_PROTOCOL,
            recovery_policy=RECOVERY_POLICY,
            required_evidence_tool_id=QA_RETRIEVAL_TOOL_ID,
        )
        outputs = {
            "retriever": _retrieval_artifact(),
            "reasoner": _reasoner_artifact(
                candidate,
                question=question,
                answer_type="city",
            ),
            "output": f"<answer>{candidate}</answer>",
        }
        execution = _execution(
            graph,
            outputs=outputs,
            final_answer=outputs["output"],
            receipt_agent_ids=("retriever",),
        )
        self.assertIsNone(env._semantic_protocol_issue(execution))

        outputs["reasoner"] = _reasoner_artifact(
            candidate,
            question=question,
            answer_type="date",
        )
        incompatible = _execution(
            graph,
            outputs=outputs,
            final_answer=outputs["output"],
            receipt_agent_ids=("retriever",),
        )
        issue = env._semantic_protocol_issue(incompatible)
        self.assertIsNotNone(issue)
        self.assertIn("answer-type constraint", issue or "")

    def test_generic_output_prompt_preserves_routed_candidate_consensus(
        self,
    ) -> None:
        registry = _registry()
        request = AgentRequest(
            request_id="hotpot:generic-output",
            run_id="hotpot",
            graph_revision=1,
            problem=SYNTHETIC_QUESTION,
            agent=_output_agent(),
            model=registry.require_model("model-b"),
            provider=registry.provider_for("model-b"),
            phase=ExecutionPhase.SINGLE,
            is_output_agent=True,
            require_exact_answer_tag=True,
            semantic_protocol=SEMANTIC_PROTOCOL,
        )
        system = build_agent_messages(request)[0]["content"]
        self.assertIn(
            "copy their agreeing candidate character-for-character",
            system,
        )
        self.assertIn("do not choose among them", system)

    def test_finish_uses_actual_routed_evidence_without_required_roles(self) -> None:
        registry = _registry()
        graph = AgentGraph(
            [_evidence_agent(), _output_agent()],
            [AgentRelation("retriever", "output", True, False)],
            output_agent_id="output",
        )
        env = _env(registry, graph=graph)
        outputs = {
            "retriever": _retrieval_artifact(),
            "output": f"<answer>{SYNTHETIC_CANDIDATE}</answer>",
        }

        valid = _execution(
            graph,
            outputs=outputs,
            final_answer=outputs["output"],
            receipt_agent_ids=("retriever",),
        )
        self.assertIsNone(env._semantic_protocol_issue(valid))

        missing_evidence = _execution(
            graph,
            outputs=outputs,
            final_answer=outputs["output"],
        )
        issue = env._semantic_protocol_issue(missing_evidence)
        self.assertIsNotNone(issue)
        self.assertIn("routed Output path has no successful", issue or "")

        unrouted_evidence = _execution(
            graph,
            outputs=outputs,
            final_answer=outputs["output"],
            receipt_agent_ids=("unrouted-agent",),
        )
        self.assertIsNotNone(env._semantic_protocol_issue(unrouted_evidence))

    def test_active_lineage_requires_the_exact_terminal_answer_syntax(self) -> None:
        registry = _registry()
        graph = AgentGraph(
            [_evidence_agent(), _output_agent()],
            [AgentRelation("retriever", "output", True, False)],
            output_agent_id="output",
        )
        env = _env(registry, graph=graph)
        outputs = {
            "retriever": _retrieval_artifact(),
            "output": _reasoner_artifact(),
        }
        unfinished = _execution(
            graph,
            outputs=outputs,
            final_answer=outputs["output"],
            receipt_agent_ids=("retriever",),
        )
        env._progressive_execution = unfinished
        env._progressive_execution_revision = graph.revision
        env._progressive_outputs = dict(outputs)
        self.assertEqual((), env._active_semantic_lineage_ids())

        outputs["output"] = f"<answer>{SYNTHETIC_CANDIDATE}</answer>"
        finished = _execution(
            graph,
            outputs=outputs,
            final_answer=outputs["output"],
            receipt_agent_ids=("retriever",),
        )
        env._progressive_execution = finished
        env._progressive_outputs = dict(outputs)
        self.assertEqual(
            ("retriever", "output"),
            env._active_semantic_lineage_ids(),
        )

    def test_selected_roles_are_validated_conditionally(self) -> None:
        registry = _registry()
        reasoner_graph = AgentGraph(
            [
                _evidence_agent(),
                AgentNode(
                    "reasoner",
                    "model-b",
                    "align routed evidence to the requested answer slot",
                    role_family="reasoner",
                    execution_mode="reasoning",
                    artifact_type="semantic_candidate",
                ),
                _output_agent(),
            ],
            [
                AgentRelation("retriever", "reasoner", True, False),
                AgentRelation("reasoner", "output", True, False),
            ],
            output_agent_id="output",
        )
        reasoner_env = _env(registry, graph=reasoner_graph)
        valid_outputs = {
            "retriever": "retrieved evidence",
            "reasoner": _reasoner_artifact(),
            "output": f"<answer>{SYNTHETIC_CANDIDATE}</answer>",
        }
        valid = _execution(
            reasoner_graph,
            outputs=valid_outputs,
            final_answer=valid_outputs["output"],
            receipt_agent_ids=("retriever",),
        )
        self.assertIsNone(reasoner_env._semantic_protocol_issue(valid))

        invalid_outputs = {**valid_outputs, "reasoner": SYNTHETIC_CANDIDATE}
        invalid = _execution(
            reasoner_graph,
            outputs=invalid_outputs,
            final_answer=valid_outputs["output"],
            receipt_agent_ids=("retriever",),
        )
        issue = reasoner_env._semantic_protocol_issue(invalid)
        self.assertIsNotNone(issue)
        self.assertIn("Reasoner 'reasoner' semantic artifact is invalid", issue or "")

        verifier_graph = AgentGraph(
            [
                _evidence_agent(),
                AgentNode(
                    "semantic_producer",
                    "model-a",
                    "determine one semantic candidate from routed evidence",
                    role_family="repair",
                    execution_mode="reasoning",
                    artifact_type="semantic_candidate",
                ),
                AgentNode(
                    "verifier",
                    "model-b",
                    "verify the routed semantic candidate",
                    role_family="verifier",
                    execution_mode="reasoning",
                    artifact_type="verification_report",
                ),
                _output_agent(),
            ],
            [
                AgentRelation("retriever", "semantic_producer", True, False),
                AgentRelation("semantic_producer", "verifier", True, False),
                AgentRelation("verifier", "output", True, False),
            ],
            output_agent_id="output",
        )
        verifier_env = _env(registry, graph=verifier_graph)
        verifier_outputs = {
            "retriever": "retrieved evidence",
            "semantic_producer": (
                f"Candidate answer: {SYNTHETIC_CANDIDATE}"
            ),
            "verifier": _verifier_artifact(),
            "output": f"<answer>{SYNTHETIC_CANDIDATE}</answer>",
        }
        verifier_execution = _execution(
            verifier_graph,
            outputs=verifier_outputs,
            final_answer=verifier_outputs["output"],
            receipt_agent_ids=("retriever",),
        )
        self.assertIsNone(verifier_env._semantic_protocol_issue(verifier_execution))

    def test_generic_output_preserves_routed_semantic_candidate(self) -> None:
        registry = _registry()
        graph = AgentGraph(
            [
                _evidence_agent(),
                AgentNode(
                    "reasoner",
                    "model-b",
                    "align routed evidence to the requested answer slot",
                    role_family="reasoner",
                    execution_mode="reasoning",
                    artifact_type="semantic_candidate",
                ),
                _output_agent(),
            ],
            [
                AgentRelation("retriever", "reasoner", True, False),
                AgentRelation("reasoner", "output", True, False),
            ],
            output_agent_id="output",
        )
        env = _env(registry, graph=graph)
        outputs = {
            "retriever": "retrieved evidence",
            "reasoner": _reasoner_artifact(),
            "output": "<answer>Other Target</answer>",
        }
        execution = _execution(
            graph,
            outputs=outputs,
            final_answer=outputs["output"],
            receipt_agent_ids=("retriever",),
        )
        issue = env._semantic_protocol_issue(execution)
        self.assertIsNotNone(issue)
        self.assertIn(
            "Generic Output Agent must preserve the routed semantic candidate",
            issue or "",
        )

    def test_generic_producer_candidate_cannot_be_changed_by_verifier(self) -> None:
        registry = _registry()
        graph = AgentGraph(
            [
                _evidence_agent(),
                AgentNode(
                    "semantic_producer",
                    "model-a",
                    "determine one semantic candidate from routed evidence",
                    role_family="repair",
                    execution_mode="reasoning",
                    artifact_type="semantic_candidate",
                ),
                AgentNode(
                    "verifier",
                    "model-b",
                    "verify the routed semantic candidate",
                    role_family="verifier",
                    execution_mode="reasoning",
                    artifact_type="verification_report",
                ),
                _output_agent(),
            ],
            [
                AgentRelation("retriever", "semantic_producer", True, False),
                AgentRelation("semantic_producer", "verifier", True, False),
                AgentRelation("verifier", "output", True, False),
            ],
            output_agent_id="output",
        )
        env = _env(registry, graph=graph)
        outputs = {
            "retriever": "retrieved evidence",
            "semantic_producer": (
                f"Candidate answer: {SYNTHETIC_CANDIDATE}"
            ),
            "verifier": _verifier_artifact("Other Target"),
            "output": "<answer>Other Target</answer>",
        }
        issue = env._semantic_protocol_issue(
            _execution(
                graph,
                outputs=outputs,
                final_answer=outputs["output"],
                receipt_agent_ids=("retriever",),
            )
        )
        self.assertIsNotNone(issue)
        self.assertIn(
            "Verifier changed a routed semantic candidate_answer",
            issue or "",
        )

    def test_reciprocal_verifiers_cannot_bootstrap_a_candidate(self) -> None:
        registry = _registry()
        graph = AgentGraph(
            [
                _evidence_agent(),
                AgentNode(
                    "context_repair",
                    "model-a",
                    "pass routed context to the checking block",
                    role_family="repair",
                    execution_mode="reasoning",
                    artifact_type="context",
                ),
                AgentNode(
                    "verifier_a",
                    "model-b",
                    "check one routed semantic candidate",
                    role_family="verifier",
                    execution_mode="reasoning",
                    artifact_type="verification_report",
                ),
                AgentNode(
                    "verifier_b",
                    "model-c",
                    "independently check one routed semantic candidate",
                    role_family="verifier",
                    execution_mode="reasoning",
                    artifact_type="verification_report",
                ),
                _output_agent(),
            ],
            [
                AgentRelation("retriever", "context_repair", True, False),
                AgentRelation("context_repair", "verifier_a", True, False),
                AgentRelation("verifier_a", "verifier_b", True, True),
                AgentRelation("verifier_b", "output", True, False),
            ],
            output_agent_id="output",
        )
        env = _env(registry, graph=graph)
        outputs = {
            "retriever": _retrieval_artifact(),
            "context_repair": "routed context without a semantic candidate",
            "verifier_a": _verifier_artifact(),
            "verifier_b": _verifier_artifact(),
            "output": f"<answer>{SYNTHETIC_CANDIDATE}</answer>",
        }
        issue = env._semantic_protocol_issue(
            _execution(
                graph,
                outputs=outputs,
                final_answer=outputs["output"],
                receipt_agent_ids=("retriever",),
            )
        )
        self.assertIsNotNone(issue)
        self.assertIn("must not bootstrap or select one", issue or "")

    def test_verifier_cannot_select_a_candidate_from_raw_retrieval(self) -> None:
        registry = _registry()
        graph = AgentGraph(
            [
                _evidence_agent(),
                AgentNode(
                    "verifier",
                    "model-b",
                    "verify one routed semantic candidate",
                    role_family="verifier",
                    execution_mode="reasoning",
                    artifact_type="verification_report",
                ),
                _output_agent(),
            ],
            [
                AgentRelation("retriever", "verifier", True, False),
                AgentRelation("verifier", "output", True, False),
            ],
            output_agent_id="output",
        )
        with self.assertRaisesRegex(
            AgentWorkflowStateError,
            "already determined semantic-candidate artifact",
        ):
            _env(registry, graph=graph)

    def test_verifier_evidence_must_be_on_its_routed_ancestor_path(self) -> None:
        registry = _registry()
        graph = AgentGraph(
            [
                _evidence_agent(),
                AgentNode(
                    "semantic_producer",
                    "model-a",
                    "determine one semantic candidate",
                    role_family="repair",
                    execution_mode="reasoning",
                    artifact_type="semantic_candidate",
                ),
                AgentNode(
                    "verifier",
                    "model-b",
                    "verify one routed semantic candidate",
                    role_family="verifier",
                    execution_mode="reasoning",
                    artifact_type="verification_report",
                ),
                _output_agent(),
            ],
            [
                AgentRelation("retriever", "output", True, False),
                AgentRelation("semantic_producer", "verifier", True, False),
                AgentRelation("verifier", "output", True, False),
            ],
            output_agent_id="output",
        )
        env = _env(registry, graph=graph)
        outputs = {
            "retriever": "retrieved evidence",
            "semantic_producer": (
                f"Candidate answer: {SYNTHETIC_CANDIDATE}"
            ),
            "verifier": _verifier_artifact(),
            "output": f"<answer>{SYNTHETIC_CANDIDATE}</answer>",
        }
        execution = _execution(
            graph,
            outputs=outputs,
            final_answer=outputs["output"],
            receipt_agent_ids=("retriever",),
        )
        env._progressive_execution = execution
        env._progressive_execution_revision = graph.revision
        env._progressive_outputs = dict(outputs)

        issue = env._semantic_protocol_issue(execution)
        self.assertIsNotNone(issue)
        self.assertIn("no routed successful qa-retrieval", issue or "")
        self.assertEqual(
            ("semantic_producer",),
            env._role_conditional_evidence_ingress_consumer_ids(),
        )
        self.assertEqual(("set_relation",), env.model_admissible_action_types())
        self.assertEqual(
            [
                {
                    "source_id": "retriever",
                    "target_id": "semantic_producer",
                    "source_to_target": True,
                    "target_to_source": False,
                }
            ],
            env._model_admissible_relation_candidates(),
        )
        repaired_graph = graph.fork()
        repaired_graph.set_relation(
            "retriever",
            "semantic_producer",
            True,
            False,
        )
        repaired_env = _env(registry, graph=repaired_graph)
        repaired_execution = _execution(
            repaired_graph,
            outputs=outputs,
            final_answer=outputs["output"],
            receipt_agent_ids=("retriever",),
        )
        repaired_env._progressive_execution = repaired_execution
        repaired_env._progressive_execution_revision = repaired_graph.revision
        repaired_env._progressive_outputs = dict(outputs)
        self.assertIsNone(
            repaired_env._semantic_protocol_issue(repaired_execution)
        )
        self.assertIs(
            repaired_env.finish_admissibility()["admissible"],
            True,
        )

    def test_generic_output_preserves_a_generic_producer_candidate(self) -> None:
        registry = _registry()
        graph = AgentGraph(
            [
                _evidence_agent(),
                AgentNode(
                    "semantic_producer",
                    "model-a",
                    "determine one semantic candidate from routed evidence",
                    role_family="repair",
                    execution_mode="reasoning",
                    artifact_type="semantic_candidate",
                ),
                _output_agent(),
            ],
            [
                AgentRelation("retriever", "semantic_producer", True, False),
                AgentRelation("semantic_producer", "output", True, False),
            ],
            output_agent_id="output",
        )
        env = _env(registry, graph=graph)
        outputs = {
            "retriever": "retrieved evidence",
            "semantic_producer": (
                f"Candidate answer: {SYNTHETIC_CANDIDATE}"
            ),
            "output": "<answer>Other Target</answer>",
        }
        issue = env._semantic_protocol_issue(
            _execution(
                graph,
                outputs=outputs,
                final_answer=outputs["output"],
                receipt_agent_ids=("retriever",),
            )
        )
        self.assertIsNotNone(issue)
        self.assertIn(
            "Generic Output Agent must preserve the routed semantic candidate",
            issue or "",
        )

    def test_generic_output_rejects_an_ambiguous_semantic_candidate_wire(
        self,
    ) -> None:
        registry = _registry()
        graph = AgentGraph(
            [
                _evidence_agent(),
                AgentNode(
                    "semantic_producer",
                    "model-a",
                    "determine one semantic candidate from routed evidence",
                    role_family="repair",
                    execution_mode="reasoning",
                    artifact_type="semantic_candidate",
                ),
                _output_agent(),
            ],
            [
                AgentRelation("retriever", "semantic_producer", True, False),
                AgentRelation("semantic_producer", "output", True, False),
            ],
            output_agent_id="output",
        )
        env = _env(registry, graph=graph)
        outputs = {
            "retriever": "retrieved evidence",
            "semantic_producer": (
                "Candidate answer: Target Gamma\nCandidate answer: Other Target"
            ),
            "output": f"<answer>{SYNTHETIC_CANDIDATE}</answer>",
        }
        issue = env._semantic_protocol_issue(
            _execution(
                graph,
                outputs=outputs,
                final_answer=outputs["output"],
                receipt_agent_ids=("retriever",),
            )
        )
        self.assertIsNotNone(issue)
        self.assertIn("semantic candidate wire must contain one", issue or "")

    def test_formatter_only_copies_a_routed_semantic_candidate(self) -> None:
        registry = _registry()
        graph = AgentGraph(
            [
                _evidence_agent(),
                AgentNode(
                    "candidate_source",
                    "model-a",
                    "determine one semantic candidate from routed evidence",
                    role_family="repair",
                    execution_mode="reasoning",
                    artifact_type="semantic_candidate",
                ),
                AgentNode(
                    "formatter",
                    "model-b",
                    _HOTPOTQA_ROLE_CONDITIONAL_FORMAT_CONTRACT,
                    role_family="format",
                    execution_mode="reasoning",
                    artifact_type="answer_wrapper",
                ),
            ],
            [
                AgentRelation("retriever", "candidate_source", True, False),
                AgentRelation("candidate_source", "formatter", True, False),
            ],
            output_agent_id="formatter",
        )
        env = _env(registry, graph=graph)
        outputs = {
            "retriever": _retrieval_artifact(),
            "candidate_source": f"Candidate answer: {SYNTHETIC_CANDIDATE}",
            "formatter": f"<answer>{SYNTHETIC_CANDIDATE}</answer>",
        }
        valid = _execution(
            graph,
            outputs=outputs,
            final_answer=outputs["formatter"],
            receipt_agent_ids=("retriever",),
        )
        self.assertIsNone(env._semantic_protocol_issue(valid))

        changed = _execution(
            graph,
            outputs={**outputs, "formatter": "<answer>Other Target</answer>"},
            final_answer="<answer>Other Target</answer>",
            receipt_agent_ids=("retriever",),
        )
        issue = env._semantic_protocol_issue(changed)
        self.assertIsNotNone(issue)
        self.assertIn("character-for-character", issue or "")

    def test_formatter_rejects_raw_retrieval_as_its_semantic_candidate(self) -> None:
        registry = _registry()
        graph = AgentGraph(
            [
                _evidence_agent(),
                AgentNode(
                    "formatter",
                    "model-b",
                    _HOTPOTQA_ROLE_CONDITIONAL_FORMAT_CONTRACT,
                    role_family="format",
                    execution_mode="reasoning",
                    artifact_type="answer_wrapper",
                ),
            ],
            [AgentRelation("retriever", "formatter", True, False)],
            output_agent_id="formatter",
        )
        with self.assertRaisesRegex(
            AgentWorkflowStateError,
            "already determined semantic-candidate artifact",
        ):
            _env(registry, graph=graph)


class RoleConditionalObservationTests(unittest.TestCase):
    def test_observation_exposes_optional_capabilities_without_fixed_spine(
        self,
    ) -> None:
        registry = _registry()
        env = _env(registry)
        orchestrator = AgentGraphOrchestrator(
            registry,
            _NoDirectorClient(),  # type: ignore[arg-type]
            tool_registry=env.runtime.tool_registry,
            prompt_version=DIRECTOR_PROMPT_VERSION,
            semantic_protocol=SEMANTIC_PROTOCOL,
            recovery_policy=RECOVERY_POLICY,
        )

        observation = orchestrator._canvas_observation(
            env,
            include_task_context=True,
            skills=(),
        )
        serialized = json.dumps(observation, sort_keys=True)

        self.assertIn("optional_role_capabilities", observation)
        for forbidden in (
            "required_direct_role_edges",
            "supported_verifier_artifact_required",
            "formatter_exact_copy_required",
            "semantic_answer_owner_count",
            "output_role_family\": \"format",
        ):
            self.assertNotIn(forbidden, serialized)
        add_domain = observation["action_target_domains"]["add_subgraph"]
        self.assertIn("output", add_domain["output_role_families"])
        self.assertIs(
            observation["terminal_constraints"]["require_format_agent"],
            False,
        )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
