"""Contracts for a role-conditional, topology-neutral HotpotQA semantic path.

The fixtures are synthetic and exercise only local prompt, Canvas, Runtime, and
terminal-admission boundaries.  They contain no benchmark answer and never call
a model or an external Tool.
"""

from __future__ import annotations

import json
import unittest

from src.interactive.agent_action_parser import parse_first_agent_action
from src.interactive.agent_graph import AgentGraph, AgentNode, AgentRelation
from src.interactive.agent_runtime import (
    AgentFailureRecord,
    AgentRequest,
    AgentRuntime,
    AgentRuntimeError,
    AgentRuntimeResult,
    ExecutionPhase,
    ReasoningExecutionAdapter,
    UpstreamMessage,
)
from src.interactive.agent_workflow_env import (
    AgentWorkflowEnv,
    _HOTPOTQA_FORMAT_CONTRACT,
)
from src.interactive.director import (
    DIRECTOR_PROMPT_VERSION,
    DIRECTOR_SYSTEM_PROMPT,
    AgentGraphOrchestrator,
)
from src.interactive.model_registry import (
    ModelRegistry,
    ModelSpec,
    ProviderSpec,
)
from src.interactive.openai_gateway import build_agent_messages
from src.interactive.qa_tool_adapter import (
    QA_RETRIEVAL_TOOL_ID,
    build_qa_tool_registry,
)


SEMANTIC_PROTOCOL = "hotpotqa_semantic_lineage_v2"
RECOVERY_POLICY = "preserve_diagnose_repair_augment"
SYNTHETIC_QUESTION = (
    "Which target is reached from Source Alpha through Bridge Beta?"
)
SYNTHETIC_CANDIDATE = "Target Gamma"


class _NoModelGateway:
    async def generate(self, request: AgentRequest) -> str:
        raise AssertionError(
            f"unit contract must not invoke Agent {request.agent.id!r}"
        )


class _NoDirectorClient:
    async def propose(self, prompt: str, **kwargs: object) -> object:
        del prompt, kwargs
        raise AssertionError("unit contract must not invoke the Director")


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
        raise AssertionError(f"unit contract must not read {passage_id!r}")


def _registry() -> ModelRegistry:
    return ModelRegistry(
        [
            ProviderSpec("provider-a", kind="test"),
            ProviderSpec("provider-b", kind="test"),
        ],
        [
            ModelSpec("model-a", "provider-a"),
            ModelSpec("model-a-spare", "provider-a"),
            ModelSpec("model-b", "provider-b"),
            ModelSpec("model-b-spare", "provider-b"),
        ],
    )


def _semantic_runtime(registry: ModelRegistry) -> AgentRuntime:
    gateway = _NoModelGateway()
    return AgentRuntime(
        registry,
        gateway,
        execution_adapters={"react": ReasoningExecutionAdapter(gateway)},
        tool_registry=build_qa_tool_registry(_NoopRetrievalIndex()),
        dataset_id="hotpotqa",
        semantic_protocol=SEMANTIC_PROTOCOL,
    )


def _semantic_env(
    registry: ModelRegistry,
    *,
    graph: AgentGraph | None = None,
) -> AgentWorkflowEnv:
    return AgentWorkflowEnv(
        registry,
        runtime=_semantic_runtime(registry),
        graph=graph,
        problem=SYNTHETIC_QUESTION,
        require_exact_answer_tag=True,
        require_format_agent=True,
        semantic_protocol=SEMANTIC_PROTOCOL,
        recovery_policy=RECOVERY_POLICY,
        required_evidence_tool_id=QA_RETRIEVAL_TOOL_ID,
    )


def _request(
    registry: ModelRegistry,
    node: AgentNode,
    *,
    upstream: tuple[UpstreamMessage, ...] = (),
    is_output_agent: bool = False,
    is_format_agent: bool = False,
    problem: str = SYNTHETIC_QUESTION,
) -> AgentRequest:
    return AgentRequest(
        request_id=f"request-{node.id}",
        run_id="synthetic-run",
        graph_revision=0,
        problem=problem,
        agent=node,
        model=registry.require_model(node.model_id),
        provider=registry.provider_for(node.model_id),
        phase=ExecutionPhase.SINGLE,
        is_output_agent=is_output_agent,
        is_format_agent=is_format_agent,
        require_exact_answer_tag=is_output_agent,
        upstream=upstream,
        semantic_protocol=SEMANTIC_PROTOCOL,
    )


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
            "multi_hop_complete": True,
            "scope_preserved": True,
            "answer_type_cardinality_correct": True,
            "minimal_answer_surface": True,
            "alias_binding_correct": True,
            "verification_status": "supported",
        },
        sort_keys=True,
    )


def _read_receipt(candidate: str = SYNTHETIC_CANDIDATE) -> dict[str, object]:
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
                        f"Bridge Beta identifies {candidate}."
                    ),
                },
            },
            "completed": True,
        },
        "error_type": None,
    }


def _single_terminal_graph() -> AgentGraph:
    return AgentGraph(
        [
            AgentNode(
                "reasoner",
                "model-a",
                "align retrieved propositions to the requested answer slot",
                role_family="reasoner",
                allowed_tools=(QA_RETRIEVAL_TOOL_ID,),
                execution_mode="react",
                artifact_type="semantic_candidate",
            ),
            AgentNode(
                "verifier",
                "model-b",
                "verify evidence, binding, hops, and scope without changing the candidate",
                role_family="verifier",
                artifact_type="verified_semantic_answer",
            ),
            AgentNode(
                "formatter",
                "model-b-spare",
                _HOTPOTQA_FORMAT_CONTRACT,
                role_family="format",
                artifact_type="answer_wrapper",
            ),
        ],
        [
            AgentRelation("reasoner", "verifier", True, False),
            AgentRelation("verifier", "formatter", True, False),
        ],
        output_agent_id="formatter",
    )


def _flexible_semantic_graph() -> AgentGraph:
    """A valid non-template graph with fan-in, parallel checks, and reciprocity."""

    return AgentGraph(
        [
            AgentNode(
                "retriever_left",
                "model-a",
                "collect one evidence branch",
                role_family="evidence_retriever",
                artifact_type="retrieval_evidence",
            ),
            AgentNode(
                "retriever_right",
                "model-b",
                "collect and cross-check another evidence branch",
                role_family="evidence_retriever",
                artifact_type="retrieval_evidence",
            ),
            AgentNode(
                "reasoner_primary",
                "model-a",
                "align evidence propositions to the requested answer slot",
                role_family="reasoner",
                allowed_tools=(QA_RETRIEVAL_TOOL_ID,),
                execution_mode="react",
                artifact_type="semantic_candidate",
            ),
            AgentNode(
                "verifier_left",
                "model-b",
                "check evidence binding and scope",
                role_family="verifier",
                artifact_type="verification_report",
            ),
            AgentNode(
                "verifier_right",
                "model-b-spare",
                "check hop completeness and answer type",
                role_family="verifier",
                artifact_type="verification_report",
            ),
            AgentNode(
                "reasoner_terminal",
                "model-a-spare",
                "resolve verification feedback and determine the semantic answer",
                role_family="reasoner",
                allowed_tools=(QA_RETRIEVAL_TOOL_ID,),
                execution_mode="react",
                artifact_type="semantic_candidate",
            ),
            AgentNode(
                "verifier_terminal",
                "model-b",
                "verify the routed semantic answer without replacing it",
                role_family="verifier",
                artifact_type="verified_semantic_answer",
            ),
            AgentNode(
                "formatter",
                "model-b-spare",
                _HOTPOTQA_FORMAT_CONTRACT,
                role_family="format",
                artifact_type="answer_wrapper",
            ),
        ],
        [
            AgentRelation("retriever_left", "retriever_right", True, True),
            AgentRelation("retriever_left", "reasoner_primary", True, False),
            AgentRelation("retriever_right", "reasoner_primary", True, False),
            AgentRelation("reasoner_primary", "verifier_left", True, False),
            AgentRelation("reasoner_primary", "verifier_right", True, False),
            AgentRelation("verifier_left", "reasoner_terminal", True, False),
            AgentRelation("verifier_right", "reasoner_terminal", True, False),
            AgentRelation("reasoner_terminal", "verifier_terminal", True, False),
            AgentRelation("verifier_terminal", "formatter", True, False),
        ],
        output_agent_id="formatter",
    )


class DirectorNeutralityTests(unittest.TestCase):
    def test_minimal_neutral_v10_does_not_prescribe_semantic_roles(self) -> None:
        self.assertEqual(
            "agentgraph.director.minimal-neutral.v10",
            DIRECTOR_PROMPT_VERSION,
        )
        prompt = DIRECTOR_SYSTEM_PROMPT.casefold()
        for forbidden in (
            "reasoner",
            "verifier",
            "formatter",
            "required_direct_role_edges",
            "semantic_answer_owner_count",
        ):
            self.assertNotIn(forbidden, prompt)

    def test_canvas_observation_exposes_capabilities_not_fixed_role_template(
        self,
    ) -> None:
        registry = _registry()
        env = _semantic_env(registry)
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

        for forbidden in (
            "required_direct_role_edges",
            "semantic_answer_owner_count",
            "verifier_count",
            "formatter_count",
            "exactly_one_reasoner",
        ):
            self.assertNotIn(forbidden, serialized)


class RoleConditionalPromptTests(unittest.TestCase):
    def test_react_is_an_execution_mode_not_a_role(self) -> None:
        registry = _registry()
        runtime = _semantic_runtime(registry)
        invalid_role = AgentNode(
            "invalid",
            "model-a",
            "collect evidence",
            role_family="react",
        )
        valid_react_reasoner = AgentNode(
            "reasoner",
            "model-a",
            "align evidence to the answer slot",
            role_family="reasoner",
            allowed_tools=(QA_RETRIEVAL_TOOL_ID,),
            execution_mode="react",
        )

        with self.assertRaisesRegex(
            AgentRuntimeError,
            "ReAct is an execution_mode, not an Agent role",
        ):
            runtime.validate_execution_contracts((invalid_role,))
        runtime.validate_execution_contracts((valid_react_reasoner,))

    def test_reasoner_and_verifier_receive_distinct_semantic_responsibilities(
        self,
    ) -> None:
        registry = _registry()
        reasoner = AgentNode(
            "reasoner",
            "model-a",
            "determine a semantic answer",
            role_family="reasoner",
        )
        verifier = AgentNode(
            "verifier",
            "model-b",
            "verify a semantic answer",
            role_family="verifier",
        )

        reasoner_system = build_agent_messages(_request(registry, reasoner))[0][
            "content"
        ].casefold()
        verifier_system = build_agent_messages(_request(registry, verifier))[0][
            "content"
        ].casefold()

        self.assertIn("subject/entity", reasoner_system)
        self.assertIn("answer slot actually requested", reasoner_system)
        self.assertIn("you alone determine the semantic candidate", reasoner_system)
        self.assertIn("original scope", reasoner_system)
        self.assertIn("explicit database or retrieved evidence", verifier_system)
        self.assertIn("entity attribute binding", verifier_system)
        self.assertIn("multi-hop", verifier_system)
        self.assertIn("original question scope", verifier_system)
        self.assertIn("must not select, replace", verifier_system)

    def test_semantic_instructions_are_applied_only_to_declared_roles(self) -> None:
        registry = _registry()
        evidence_agent = AgentNode(
            "evidence",
            "model-a",
            "return a grounded evidence artifact",
            role_family="evidence_retriever",
        )

        system = build_agent_messages(_request(registry, evidence_agent))[0][
            "content"
        ].casefold()

        self.assertNotIn("you alone determine the semantic candidate", system)
        self.assertNotIn("you are the semantic verifier", system)

    def test_formatter_receives_only_verified_artifact_and_copies_candidate(
        self,
    ) -> None:
        registry = _registry()
        original_question = "ORIGINAL_SCOPE_SENTINEL must never reach Formatter"
        formatter = AgentNode(
            "formatter",
            "model-b-spare",
            _HOTPOTQA_FORMAT_CONTRACT,
            role_family="format",
        )
        upstream = UpstreamMessage(
            source_agent_id="verifier_terminal",
            target_agent_id="formatter",
            content=_verifier_artifact(),
            artifact_type="verified_semantic_answer",
        )

        messages = build_agent_messages(
            _request(
                registry,
                formatter,
                upstream=(upstream,),
                is_output_agent=True,
                is_format_agent=True,
                problem=original_question,
            )
        )
        rendered = "\n".join(message["content"] for message in messages)

        self.assertNotIn(original_question, rendered)
        self.assertIn(SYNTHETIC_CANDIDATE, rendered)
        self.assertIn("never the original question", rendered)
        self.assertIn("do not solve, reason, verify", rendered)
        self.assertIn("Copy character-for-character", rendered)
        self.assertIn("never select another name or value", rendered)


class FlexibleSemanticGraphTests(unittest.TestCase):
    def test_semantic_protocol_does_not_synthesize_required_role_edges(self) -> None:
        registry = _registry()
        env = _semantic_env(registry, graph=_single_terminal_graph())

        self.assertEqual((), env._required_semantic_edges())
        self.assertEqual([], env._required_semantic_relation_candidates())

    def test_multiple_semantic_roles_fanin_and_reciprocity_are_admissible(
        self,
    ) -> None:
        registry = _registry()
        graph = _flexible_semantic_graph()

        validation = graph.validate(registry, require_complete=True)
        self.assertTrue(validation.valid, validation.issues)
        env = _semantic_env(registry, graph=graph)
        self.assertIsNone(env._semantic_edit_issue_for(graph))
        self.assertEqual(
            ("retriever_left", "retriever_right"),
            graph.directed_predecessors("reasoner_primary"),
        )
        role_counts: dict[str, int] = {}
        for node in graph.nodes:
            role = node.role_family or ""
            role_counts[role] = role_counts.get(role, 0) + 1
        self.assertGreaterEqual(role_counts["reasoner"], 2)
        self.assertGreaterEqual(role_counts["verifier"], 2)
        motifs = graph.topology_statistics()["topology_motifs"]
        self.assertIn("fan_in", motifs)
        self.assertIn("reciprocal", motifs)

    def test_output_is_not_exposed_while_a_branch_cannot_reach_it(self) -> None:
        registry = _registry()
        graph = _single_terminal_graph()
        graph = AgentGraph(graph.nodes, graph.relations)
        graph.add_agent(
            AgentNode(
                "orphan_reasoner",
                "model-a-spare",
                "independently check one semantic interpretation",
                role_family="reasoner",
                execution_mode="reasoning",
                artifact_type="semantic_candidate",
            )
        )
        env = _semantic_env(registry, graph=graph)

        self.assertEqual((), env._model_admissible_output_agent_ids())
        candidate = graph.fork()
        candidate.set_output("formatter")
        issue = env._semantic_edit_issue_for(candidate)
        self.assertIsNotNone(issue)
        self.assertIn("terminal_unreachable_agent_ids=['orphan_reasoner']", issue or "")

    def test_output_is_exposed_after_all_branches_are_routed(self) -> None:
        registry = _registry()
        complete = _flexible_semantic_graph()
        graph = AgentGraph(complete.nodes, complete.relations)
        env = _semantic_env(registry, graph=graph)

        self.assertEqual(("formatter",), env._model_admissible_output_agent_ids())
        candidate = graph.fork()
        candidate.set_output("formatter")
        self.assertIsNone(env._semantic_edit_issue_for(candidate))

    def test_output_revision_rejects_a_later_isolated_branch(self) -> None:
        registry = _registry()
        graph = _single_terminal_graph()
        env = _semantic_env(registry, graph=graph)
        candidate = graph.fork()
        candidate.add_agent(
            AgentNode(
                "late_orphan",
                "model-a-spare",
                "check an additional interpretation",
                role_family="reasoner",
                execution_mode="reasoning",
                artifact_type="semantic_candidate",
            )
        )

        issue = env._semantic_edit_issue_for(candidate)
        self.assertIsNotNone(issue)
        self.assertIn("terminal_unreachable_agent_ids=['late_orphan']", issue or "")

        candidate.set_relation("late_orphan", "verifier", True, False)
        self.assertIsNone(env._semantic_edit_issue_for(candidate))

    def test_terminal_gate_compares_actual_routed_artifacts(self) -> None:
        registry = _registry()
        graph = _flexible_semantic_graph()
        env = _semantic_env(registry, graph=graph)
        outputs = {
            "retriever_left": "synthetic evidence branch left",
            "retriever_right": "synthetic evidence branch right",
            "reasoner_primary": "non-terminal semantic artifact",
            "verifier_left": "non-terminal verification artifact left",
            "verifier_right": "non-terminal verification artifact right",
            "reasoner_terminal": _reasoner_artifact(),
            "verifier_terminal": _verifier_artifact(),
            "formatter": f"<answer>{SYNTHETIC_CANDIDATE}</answer>",
        }
        execution = AgentRuntimeResult(
            run_id="synthetic-run",
            graph_revision=graph.revision,
            output_agent_id="formatter",
            final_answer=outputs["formatter"],
            outputs=outputs,
            calls=(),
            block_completion_order=(),
            output_metadata={
                "reasoner_terminal": {"tool_receipts": [_read_receipt()]}
            },
        )

        self.assertIsNone(env._semantic_protocol_issue(execution))

        verifier_mismatch_outputs = dict(outputs)
        verifier_mismatch_outputs["verifier_terminal"] = _verifier_artifact(
            "Different Synthetic Target"
        )
        verifier_mismatch = AgentRuntimeResult(
            run_id="synthetic-run-verifier-mismatch",
            graph_revision=graph.revision,
            output_agent_id="formatter",
            final_answer=outputs["formatter"],
            outputs=verifier_mismatch_outputs,
            calls=(),
            block_completion_order=(),
            output_metadata=execution.output_metadata,
        )
        verifier_issue = env._semantic_protocol_issue(verifier_mismatch)
        self.assertIsNotNone(verifier_issue)
        self.assertIn("Verifier changed", verifier_issue or "")

        formatter_mismatch = AgentRuntimeResult(
            run_id="synthetic-run-formatter-mismatch",
            graph_revision=graph.revision,
            output_agent_id="formatter",
            final_answer="<answer>Different Synthetic Target</answer>",
            outputs={
                **outputs,
                "formatter": "<answer>Different Synthetic Target</answer>",
            },
            calls=(),
            block_completion_order=(),
            output_metadata=execution.output_metadata,
        )
        formatter_issue = env._semantic_protocol_issue(formatter_mismatch)
        self.assertIsNotNone(formatter_issue)
        self.assertIn("Formatter must only wrap", formatter_issue or "")


class PreserveRecoveryTests(unittest.TestCase):
    def test_provider_repair_precedes_topology_edits_and_changes_only_model(
        self,
    ) -> None:
        registry = _registry()
        graph = AgentGraph(
            [AgentNode("worker", "model-a", "produce an intermediate artifact")],
            output_agent_id="worker",
        )
        env = AgentWorkflowEnv(
            registry,
            _NoModelGateway(),
            graph=graph,
            problem="synthetic task",
            recovery_policy=RECOVERY_POLICY,
        )
        failure = AgentFailureRecord(
            request_id="provider-failure",
            agent_id="worker",
            phase=ExecutionPhase.SINGLE,
            graph_revision=graph.revision,
            error_type="OpenAICompatibleGatewayError",
            message="provider request failed with HTTP status 429",
        )
        env._record_failure_state((failure,), current_agent_ids={"worker"})

        self.assertEqual(("modify_agent",), env.model_admissible_action_types())
        targets = env.model_admissible_action_targets()
        self.assertIn("modify_agent", targets)
        for disallowed in ("add_subgraph", "delete_agent", "set_relation"):
            self.assertNotIn(disallowed, targets)
        modify = targets["modify_agent"]
        self.assertEqual(["worker"], modify["agent_ids"])
        candidate = modify["per_agent_candidates"][0]
        self.assertEqual(["model_id"], candidate["mutable_fields"])
        self.assertEqual(
            ["model-b", "model-b-spare"],
            candidate["discrete_value_domains"]["model_id"],
        )

        contract_edit = parse_first_agent_action(
            '{"action":"modify_agent","agent_id":"worker",'
            '"contract":"discard the current artifact"}'
        )
        self.assertIn(
            "must modify only model_id",
            env._provider_repair_admission_issue(contract_edit) or "",
        )
        model_repair = parse_first_agent_action(
            '{"action":"modify_agent","agent_id":"worker",'
            '"model_id":"model-b"}'
        )
        self.assertIsNone(env._provider_repair_admission_issue(model_repair))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
