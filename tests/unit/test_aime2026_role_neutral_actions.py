from __future__ import annotations

import json
import unittest

from src.interactive.agent_graph import AgentGraph, AgentNode
from src.interactive.agent_runtime import (
    AgentFailureRecord,
    AgentRequest,
    AgentRuntime,
    AgentRuntimeError,
    ExecutionPhase,
    ReasoningExecutionAdapter,
)
from src.interactive.agent_workflow_env import AgentWorkflowEnv
from src.interactive.director import (
    HOTPOTQA_SEMANTIC_PROTOCOL,
    PRESERVE_DIAGNOSE_REPAIR_AUGMENT_POLICY,
    QA_VERIFIED_ANSWER_LINEAGE_PROTOCOL,
    director_live_add_subgraph_agent_declarations_from_text,
    director_live_add_subgraph_agent_declarations_json_schema_text,
)
from src.interactive.model_registry import ModelRegistry, ModelSpec, ProviderSpec
from src.interactive.qa_tool_adapter import (
    QA_RETRIEVAL_TOOL_ID,
    build_qa_tool_registry,
)


class _RecordingGateway:
    def __init__(self) -> None:
        self.requests: list[AgentRequest] = []

    async def generate(self, request: AgentRequest) -> str:
        self.requests.append(request)
        return f"answer:{request.agent.id}"


class _CountingRuntime(AgentRuntime):
    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)
        self.execute_calls = 0

    async def execute(self, *args: object, **kwargs: object):  # type: ignore[no-untyped-def]
        self.execute_calls += 1
        return await super().execute(*args, **kwargs)


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
        raise AssertionError(f"unexpected retrieval read {passage_id!r}")


def _registry() -> ModelRegistry:
    return ModelRegistry(
        [ProviderSpec("test-provider", kind="test")],
        [ModelSpec("test-model", "test-provider")],
    )


def _multi_provider_registry() -> ModelRegistry:
    return ModelRegistry(
        [
            ProviderSpec("provider-a", kind="test"),
            ProviderSpec("provider-b", kind="test"),
        ],
        [
            ModelSpec("model-a", "provider-a"),
            ModelSpec("model-b", "provider-b"),
        ],
    )


def _aime_env(
    *,
    graph: AgentGraph | None = None,
    require_format_agent: bool = False,
    explicit_none: bool = True,
    execute_on_edit: bool = True,
) -> tuple[AgentWorkflowEnv, _CountingRuntime, _RecordingGateway]:
    registry = _registry()
    gateway = _RecordingGateway()
    runtime = _CountingRuntime(
        registry,
        gateway,
        dataset_id="aime_2026",
        **({"semantic_protocol": "none"} if explicit_none else {}),
    )
    env = AgentWorkflowEnv(
        registry,
        runtime=runtime,
        problem="Find the requested AIME integer.",
        graph=graph,
        execute_on_edit=execute_on_edit,
        require_format_agent=require_format_agent,
        **({"semantic_protocol": "none"} if explicit_none else {}),
    )
    return env, runtime, gateway


def _qa_env(
    *,
    dataset_id: str,
    semantic_protocol: str,
) -> AgentWorkflowEnv:
    registry = _registry()
    gateway = _RecordingGateway()
    runtime = _CountingRuntime(
        registry,
        gateway,
        dataset_id=dataset_id,
        semantic_protocol=semantic_protocol,
        execution_adapters={"react": ReasoningExecutionAdapter(gateway)},
        tool_registry=build_qa_tool_registry(_NoopRetrievalIndex()),
    )
    return AgentWorkflowEnv(
        registry,
        runtime=runtime,
        problem="Which entity satisfies the requested relation?",
        semantic_protocol=semantic_protocol,
        recovery_policy=PRESERVE_DIAGNOSE_REPAIR_AUGMENT_POLICY,
        required_evidence_tool_id=QA_RETRIEVAL_TOOL_ID,
    )


def _agent_branches(schema: dict[str, object]) -> tuple[dict[str, object], ...]:
    properties = schema["properties"]
    assert isinstance(properties, dict)
    agents = properties["agents"]
    assert isinstance(agents, dict)
    count_branches = agents["oneOf"]
    assert isinstance(count_branches, list)
    result: list[dict[str, object]] = []
    for count_branch in count_branches:
        assert isinstance(count_branch, dict)
        positions = count_branch["prefixItems"]
        assert isinstance(positions, list)
        for position in positions:
            assert isinstance(position, dict)
            branches = position["anyOf"]
            assert isinstance(branches, list)
            result.extend(branch for branch in branches if isinstance(branch, dict))
    return tuple(result)


class AIME2026RoleNeutralActionTests(unittest.IsolatedAsyncioTestCase):
    def test_generic_live_declaration_schema_and_parser_reject_role_family(
        self,
    ) -> None:
        env, runtime, gateway = _aime_env()
        domains = env.model_admissible_action_targets()
        add_domain = domains["add_subgraph"]
        self.assertIsInstance(add_domain, dict)
        self.assertNotIn("role_family_admitted", add_domain)
        self.assertNotIn("role_family", add_domain["required_agent_fields"])

        schema = json.loads(
            director_live_add_subgraph_agent_declarations_json_schema_text(
                domains
            )
        )
        branches = _agent_branches(schema)
        self.assertTrue(branches)
        for branch in branches:
            self.assertNotIn("role_family", branch["properties"])
            self.assertNotIn("role_family", branch["required"])

        valid_agent = {
            "agent_id": "node_1",
            "model_id": "test-model",
            "contract": "derive a result from the original problem",
            "execution_mode": "reasoning",
            "allowed_tools": [],
        }
        parsed = director_live_add_subgraph_agent_declarations_from_text(
            json.dumps({"action": "add_subgraph", "agents": [valid_agent]}),
            domains,
        )
        self.assertEqual((valid_agent,), parsed)

        with self.assertRaisesRegex(ValueError, "fields are invalid"):
            director_live_add_subgraph_agent_declarations_from_text(
                json.dumps(
                    {
                        "action": "add_subgraph",
                        "agents": [
                            {**valid_agent, "role_family": "calculator_agent"}
                        ],
                    }
                ),
                domains,
            )
        self.assertEqual(0, runtime.execute_calls)
        self.assertEqual([], gateway.requests)

    async def test_role_neutral_canvas_fails_closed_for_all_agent_edits(
        self,
    ) -> None:
        cases = (
            (
                "default-none-add-agent",
                False,
                None,
                {
                    "action": "add_agent",
                    "agent_id": "node_1",
                    "model_id": "test-model",
                    "contract": "derive a result",
                    "role_family": "solver",
                    "execution_mode": "reasoning",
                    "allowed_tools": [],
                },
            ),
            (
                "explicit-none-add-subgraph",
                True,
                None,
                {
                    "action": "add_subgraph",
                    "agents": [
                        {
                            "agent_id": "node_1",
                            "model_id": "test-model",
                            "contract": "derive a result",
                            "role_family": "calculator_agent",
                            "execution_mode": "reasoning",
                            "allowed_tools": [],
                        }
                    ],
                    "relations": [],
                },
            ),
            (
                "explicit-none-modify-agent",
                True,
                AgentGraph(
                    [AgentNode("node_1", "test-model", "derive a result")]
                ),
                {
                    "action": "modify_agent",
                    "agent_id": "node_1",
                    "role_family": "verifier",
                },
            ),
        )
        for label, explicit_none, graph, payload in cases:
            with self.subTest(label=label):
                env, runtime, gateway = _aime_env(
                    graph=graph,
                    explicit_none=explicit_none,
                )
                revision = env.graph.revision
                nodes = env.graph.nodes

                result = await env.step(json.dumps(payload))

                self.assertFalse(result.accepted)
                self.assertEqual("role_family_forbidden", result.feedback_code)
                self.assertIn("role_family", result.feedback)
                self.assertEqual(revision, env.graph.revision)
                self.assertEqual(nodes, env.graph.nodes)
                self.assertEqual(0, runtime.execute_calls)
                self.assertEqual([], gateway.requests)

    async def test_legacy_format_agent_boundary_still_allows_role_family(
        self,
    ) -> None:
        env, runtime, gateway = _aime_env(
            require_format_agent=True,
            execute_on_edit=False,
        )
        targets = env.model_admissible_action_targets()
        self.assertIs(targets["add_subgraph"]["role_family_admitted"], True)

        result = await env.step(
            json.dumps(
                {
                    "action": "add_subgraph",
                    "agents": [
                        {
                            "agent_id": "node_1",
                            "model_id": "test-model",
                            "contract": "derive one semantic answer",
                            "role_family": "reasoner",
                            "execution_mode": "reasoning",
                            "allowed_tools": [],
                        },
                        {
                            "agent_id": "node_2",
                            "model_id": "test-model",
                            "contract": (
                                "copy one routed result into the output wrapper"
                            ),
                            "role_family": "format",
                            "execution_mode": "reasoning",
                            "allowed_tools": [],
                        },
                    ],
                    "relations": [
                        {
                            "source_id": "node_1",
                            "target_id": "node_2",
                            "source_to_target": True,
                            "target_to_source": False,
                        }
                    ],
                    "output_agent_id": "node_2",
                }
            )
        )

        self.assertTrue(result.accepted, result.feedback)
        self.assertEqual("reasoner", env.graph.get_node("node_1").role_family)
        self.assertEqual("format", env.graph.get_node("node_2").role_family)
        self.assertEqual(0, runtime.execute_calls)
        self.assertEqual([], gateway.requests)

    async def test_non_aime_generic_canvas_preserves_legacy_role_metadata(
        self,
    ) -> None:
        registry = _registry()
        gateway = _RecordingGateway()
        runtime = _CountingRuntime(registry, gateway, semantic_protocol="none")
        env = AgentWorkflowEnv(
            registry,
            runtime=runtime,
            problem="Complete the generic task.",
            execute_on_edit=False,
            semantic_protocol="none",
            require_format_agent=False,
        )

        result = await env.step(
            json.dumps(
                {
                    "action": "add_agent",
                    "agent_id": "node_1",
                    "model_id": "test-model",
                    "contract": "complete the requested task",
                    "role_family": "legacy-metadata",
                    "execution_mode": "reasoning",
                    "allowed_tools": [],
                }
            )
        )

        self.assertTrue(result.accepted, result.feedback)
        self.assertEqual(
            "legacy-metadata", env.graph.get_node("node_1").role_family
        )
        self.assertEqual(0, runtime.execute_calls)
        self.assertEqual([], gateway.requests)

    def test_hotpotqa_and_triviaqa_role_conditioned_paths_remain_admitted(
        self,
    ) -> None:
        cases = (
            (
                "hotpotqa",
                HOTPOTQA_SEMANTIC_PROTOCOL,
                "react",
                [QA_RETRIEVAL_TOOL_ID],
            ),
            (
                "triviaqa",
                QA_VERIFIED_ANSWER_LINEAGE_PROTOCOL,
                "reasoning",
                [],
            ),
        )
        for dataset_id, semantic_protocol, execution_mode, allowed_tools in cases:
            with self.subTest(dataset_id=dataset_id):
                env = _qa_env(
                    dataset_id=dataset_id,
                    semantic_protocol=semantic_protocol,
                )
                domains = env.model_admissible_action_targets()
                add_domain = domains["add_subgraph"]
                self.assertIn("role_family", add_domain["required_agent_fields"])
                self.assertIn("reasoner", add_domain["role_constraints"])

                schema = json.loads(
                    director_live_add_subgraph_agent_declarations_json_schema_text(
                        domains
                    )
                )
                branches = _agent_branches(schema)
                self.assertTrue(branches)
                self.assertTrue(
                    all("role_family" in branch["required"] for branch in branches)
                )
                self.assertTrue(
                    all(
                        "role_family" in branch["properties"]
                        for branch in branches
                    )
                )

                declaration = {
                    "agent_id": "node_1",
                    "model_id": "test-model",
                    "contract": (
                        "bind public evidence to the requested relation and "
                        "derive one semantic candidate"
                    ),
                    "role_family": "reasoner",
                    "execution_mode": execution_mode,
                    "allowed_tools": allowed_tools,
                }
                parsed = director_live_add_subgraph_agent_declarations_from_text(
                    json.dumps(
                        {"action": "add_subgraph", "agents": [declaration]}
                    ),
                    domains,
                )
                self.assertEqual((declaration,), parsed)

    async def test_aime_model_visible_artifacts_omit_role_metadata(self) -> None:
        env, _, _ = _aime_env()
        result = await env.step(
            json.dumps(
                {
                    "action": "add_subgraph",
                    "agents": [
                        {
                            "agent_id": "node_1",
                            "model_id": "test-model",
                            "contract": "derive a result from the problem",
                            "execution_mode": "reasoning",
                            "allowed_tools": [],
                        }
                    ],
                    "relations": [],
                    "output_agent_id": "node_1",
                }
            )
        )

        self.assertTrue(result.accepted, result.feedback)
        payload = json.loads(result.feedback.split("execution_result=", 1)[1])
        self.assertEqual(1, len(payload["agent_artifacts"]))
        self.assertNotIn("role_family", payload["agent_artifacts"][0])
        self.assertNotIn("execution_role", payload["agent_artifacts"][0])
        self.assertNotIn(
            "role_family",
            json.dumps(env.model_admissible_action_targets()),
        )

    def test_aime_recovery_feedback_omits_role_parameter(self) -> None:
        registry = _multi_provider_registry()
        gateway = _RecordingGateway()
        runtime = AgentRuntime(
            registry,
            gateway,
            dataset_id="aime_2026",
            semantic_protocol="none",
        )
        graph = AgentGraph(
            [AgentNode("node_1", "model-a", "derive a result")]
        )
        env = AgentWorkflowEnv(
            registry,
            runtime=runtime,
            graph=graph,
            problem="Find the requested AIME integer.",
            semantic_protocol="none",
            recovery_policy=PRESERVE_DIAGNOSE_REPAIR_AUGMENT_POLICY,
        )
        failures = (
            AgentFailureRecord(
                request_id="provider-timeout",
                agent_id="node_1",
                phase=ExecutionPhase.SINGLE,
                graph_revision=graph.revision,
                error_type="TimeoutError",
                message="provider request timed out",
            ),
            AgentFailureRecord(
                request_id="structured-action-invalid",
                agent_id="node_1",
                phase=ExecutionPhase.SINGLE,
                graph_revision=graph.revision,
                error_type="ReactExecutionError",
                message=(
                    "StructuredAction serialization remained invalid after one "
                    "bounded same-model regeneration"
                ),
                metadata={
                    "failure_category": (
                        "structured_action_serialization_failure"
                    ),
                    "react_trace": [
                        {
                            "observation_status": "parse_error",
                            "public_error_code": "structured_action_truncated",
                        }
                    ],
                },
            ),
        )
        for failure in failures:
            with self.subTest(error_type=failure.error_type):
                feedback = json.loads(
                    env._execution_error_feedback(
                        AgentRuntimeError(
                            "execution failed",
                            failure_records=(failure,),
                        )
                    ).split("=", 1)[1]
                )
                preferred = feedback["failed_agents"][0]["preferred_repair"]
                self.assertNotIn("role_family", preferred["preserve_fields"])

    def test_legacy_role_protocol_keeps_role_in_recovery_feedback(self) -> None:
        graph = AgentGraph(
            [
                AgentNode(
                    "node_1",
                    "test-model",
                    "derive one semantic result",
                    role_family="reasoner",
                )
            ]
        )
        env, _, _ = _aime_env(
            graph=graph,
            require_format_agent=True,
            execute_on_edit=False,
        )
        failure = AgentFailureRecord(
            request_id="structured-action-invalid",
            agent_id="node_1",
            phase=ExecutionPhase.SINGLE,
            graph_revision=graph.revision,
            error_type="ReactExecutionError",
            message="StructuredAction serialization remained invalid",
            metadata={
                "failure_category": "structured_action_serialization_failure"
            },
        )
        feedback = json.loads(
            env._execution_error_feedback(
                AgentRuntimeError(
                    "execution failed",
                    failure_records=(failure,),
                )
            ).split("=", 1)[1]
        )
        preferred = feedback["failed_agents"][0]["preferred_repair"]
        self.assertIn("role_family", preferred["preserve_fields"])


if __name__ == "__main__":
    unittest.main()
