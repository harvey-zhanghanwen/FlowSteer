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
    AgentRequest,
    AgentRuntime,
    AgentRuntimeResult,
    ExecutionPhase,
    ReasoningExecutionAdapter,
)
from src.interactive.agent_workflow_env import (
    AgentWorkflowEnv,
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
) -> AgentWorkflowEnv:
    return AgentWorkflowEnv(
        registry,
        runtime=_runtime(registry),
        graph=graph,
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


def _reasoner_artifact(candidate: str = SYNTHETIC_CANDIDATE) -> str:
    return json.dumps(
        {
            "question_scope": SYNTHETIC_QUESTION,
            "answer_slot": {
                "answer_type": "entity",
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

    def test_add_subgraph_can_atomically_handoff_an_unfinished_output(self) -> None:
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
            output_agent_id="reasoner",
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
        self.assertIn("output_agent_id", schema["required"])
        output_enums = set(output_schema["enum"])
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
        self.assertIn("routed semantic consumer", issue or "")

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
        declaration = {
            "agent_id": "node_1",
            "model_id": "model-b",
            "contract": "retrieve explicit evidence for the original question",
            "role_family": "evidence_retriever",
            "allowed_tools": [QA_RETRIEVAL_TOOL_ID],
            "execution_mode": "react",
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

    def test_materialized_existing_agent_repairs_deferred_consumer_by_relation(
        self,
    ) -> None:
        registry = _registry()
        graph = AgentGraph(
            [
                _evidence_agent("retriever"),
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
            outputs={"retriever": "retrieved evidence"},
            calls=(),
            block_completion_order=(),
            executed_agent_ids=("retriever",),
            deferred_agent_ids=("verifier",),
        )
        env._progressive_execution_revision = graph.revision

        self.assertEqual(("set_relation",), env.model_admissible_action_types())
        candidates = env.model_admissible_action_targets()["set_relation"][
            "candidates"
        ]
        self.assertTrue(candidates)
        for relation in candidates:
            self.assertEqual("retriever", relation["source_id"])
            self.assertEqual("verifier", relation["target_id"])
            self.assertIs(relation["source_to_target"], True)


class RoleConditionalTerminalTests(unittest.TestCase):
    def test_finish_uses_actual_routed_evidence_without_required_roles(self) -> None:
        registry = _registry()
        graph = AgentGraph(
            [_evidence_agent(), _output_agent()],
            [AgentRelation("retriever", "output", True, False)],
            output_agent_id="output",
        )
        env = _env(registry, graph=graph)
        outputs = {
            "retriever": f"Candidate answer: {SYNTHETIC_CANDIDATE}",
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
            "retriever": f"Candidate answer: {SYNTHETIC_CANDIDATE}",
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
                AgentRelation("retriever", "verifier", True, False),
                AgentRelation("verifier", "output", True, False),
            ],
            output_agent_id="output",
        )
        verifier_env = _env(registry, graph=verifier_graph)
        verifier_outputs = {
            "retriever": f"Candidate answer: {SYNTHETIC_CANDIDATE}",
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

    def test_formatter_only_copies_a_routed_semantic_candidate(self) -> None:
        registry = _registry()
        graph = AgentGraph(
            [
                _evidence_agent("candidate_source"),
                AgentNode(
                    "formatter",
                    "model-b",
                    _HOTPOTQA_ROLE_CONDITIONAL_FORMAT_CONTRACT,
                    role_family="format",
                    execution_mode="reasoning",
                    artifact_type="answer_wrapper",
                ),
            ],
            [AgentRelation("candidate_source", "formatter", True, False)],
            output_agent_id="formatter",
        )
        env = _env(registry, graph=graph)
        outputs = {
            "candidate_source": f"Candidate answer: {SYNTHETIC_CANDIDATE}",
            "formatter": f"<answer>{SYNTHETIC_CANDIDATE}</answer>",
        }
        valid = _execution(
            graph,
            outputs=outputs,
            final_answer=outputs["formatter"],
            receipt_agent_ids=("candidate_source",),
        )
        self.assertIsNone(env._semantic_protocol_issue(valid))

        changed = _execution(
            graph,
            outputs={**outputs, "formatter": "<answer>Other Target</answer>"},
            final_answer="<answer>Other Target</answer>",
            receipt_agent_ids=("candidate_source",),
        )
        issue = env._semantic_protocol_issue(changed)
        self.assertIsNotNone(issue)
        self.assertIn("character-for-character", issue or "")


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
