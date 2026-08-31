from __future__ import annotations

import json
import unittest

from src.interactive.agent_graph import AgentGraph, AgentNode, AgentRelation
from src.interactive.agent_runtime import AgentFailureRecord, ExecutionPhase
from src.interactive.agent_workflow_env import AgentWorkflowEnv
from src.interactive.director import (
    director_live_action_parameter_json_schema_text,
)
from src.interactive.model_registry import ModelRegistry, ModelSpec, ProviderSpec


def _registry() -> ModelRegistry:
    return ModelRegistry(
        [ProviderSpec("provider", kind="test")],
        [ModelSpec("model", "provider")],
    )


def _node(agent_id: str) -> AgentNode:
    return AgentNode(
        agent_id,
        "model",
        (
            "Use the supplied conversation and routed artifacts to produce "
            "a useful artifact."
        ),
    )


def _env(
    graph: AgentGraph,
    *,
    max_agents: int = 2,
    recovery_policy: str = "default",
) -> AgentWorkflowEnv:
    env = AgentWorkflowEnv(
        _registry(),
        gateway=object(),
        graph=graph,
        max_agents=max_agents,
        max_agents_per_subgraph=3,
        max_relations_per_subgraph=2,
        allowed_actions=(
            "add_subgraph",
            "modify_agent",
            "delete_agent",
            "set_relation",
            "set_output",
            "finish",
        ),
        finish_only_when_admissible=True,
        require_output_protocol_artifact_for_set_output=True,
        recovery_policy=recovery_policy,
    )
    env.reset(
        "A healthcare conversation requiring a complete assistant response",
        graph,
    )
    return env


def _seed_artifact(
    env: AgentWorkflowEnv,
    agent_id: str,
    *,
    generated_as_output_agent: bool,
) -> None:
    env._progressive_outputs[agent_id] = "Materialized response artifact"
    env._progressive_output_metadata[agent_id] = {
        "artifact_version": f"{agent_id}:revision-1",
        "generated_as_output_agent": generated_as_output_agent,
        "input_artifact_versions": {},
    }


def _add_action(*, output_agent_id: str | None) -> str:
    payload: dict[str, object] = {
        "action": "add_subgraph",
        "agents": [
            {
                "agent_id": "node_2",
                "model_id": "model",
                "contract": (
                    "Consume the routed artifact and produce one complete "
                    "user-facing assistant response."
                ),
                "execution_mode": "reasoning",
                "allowed_tools": [],
            }
        ],
        "relations": [
            {
                "source_id": "node_1",
                "target_id": "node_2",
                "source_to_target": True,
                "target_to_source": False,
            }
        ],
    }
    if output_agent_id is not None:
        payload["output_agent_id"] = output_agent_id
    return json.dumps(payload)


class HealthBenchOutputClosureDomainTests(unittest.TestCase):
    def test_intermediate_only_state_exposes_one_atomic_output_closure(self) -> None:
        graph = AgentGraph([_node("node_1")])
        env = _env(graph)
        _seed_artifact(env, "node_1", generated_as_output_agent=False)

        self.assertEqual(("add_subgraph",), env.model_admissible_action_types())
        domain = env.model_admissible_action_targets()["add_subgraph"]
        self.assertEqual(1, domain["min_new_agents"])
        self.assertEqual(1, domain["max_new_agents"])
        self.assertEqual(
            {
                "mode": "required_new_terminal_consumer",
                "eligible_existing_agent_ids": [],
                "same_action_agents_eligible": True,
                "remaining_capacity": 1,
            },
            {
                key: domain["output_provenance"][key]
                for key in (
                    "mode",
                    "eligible_existing_agent_ids",
                    "same_action_agents_eligible",
                    "remaining_capacity",
                )
            },
        )

    def test_final_add_schema_requires_same_action_output_agent(self) -> None:
        graph = AgentGraph([_node("node_1")])
        env = _env(graph)
        _seed_artifact(env, "node_1", generated_as_output_agent=False)
        domains = env.model_admissible_action_targets()
        declarations = [
            {
                "agent_id": "node_2",
                "model_id": "model",
                "contract": (
                    "Consume the routed artifact and produce one complete "
                    "user-facing assistant response."
                ),
                "execution_mode": "reasoning",
                "allowed_tools": [],
            }
        ]

        schema = json.loads(
            director_live_action_parameter_json_schema_text(
                "add_subgraph",
                domains,
                add_agents=declarations,
            )
        )

        self.assertIn("output_agent_id", schema["required"])
        self.assertEqual(
            {"const": "node_2"},
            schema["properties"]["output_agent_id"],
        )

    def test_output_provenance_is_optional_with_existing_safe_target(self) -> None:
        graph = AgentGraph([_node("node_1")])
        env = _env(graph)
        _seed_artifact(env, "node_1", generated_as_output_agent=True)

        receipt = env.model_admissible_action_targets()["add_subgraph"][
            "output_provenance"
        ]
        self.assertEqual("optional", receipt["mode"])
        self.assertEqual(["node_1"], receipt["eligible_existing_agent_ids"])
        self.assertEqual(1, receipt["remaining_capacity"])

    def test_finite_empty_canvas_protects_the_capacity_exhausting_add(self) -> None:
        graph = AgentGraph()
        env = _env(graph)

        receipt = env.model_admissible_action_targets()["add_subgraph"][
            "output_provenance"
        ]
        self.assertEqual("require_if_capacity_exhausted", receipt["mode"])
        self.assertEqual([], receipt["eligible_existing_agent_ids"])
        self.assertTrue(receipt["same_action_agents_eligible"])
        self.assertEqual(2, receipt["remaining_capacity"])

    def test_closure_ingress_is_restricted_to_quotient_sink_artifacts(self) -> None:
        graph = AgentGraph(
            [_node("node_1"), _node("node_2"), _node("node_3")],
            [
                AgentRelation("node_1", "node_2", True, False),
                AgentRelation("node_3", "node_2", True, False),
            ],
        )
        env = _env(graph, max_agents=4)
        for agent_id in ("node_1", "node_2", "node_3"):
            _seed_artifact(
                env,
                agent_id,
                generated_as_output_agent=False,
            )

        receipt = env.model_admissible_action_targets()["add_subgraph"][
            "output_provenance"
        ]

        self.assertEqual("required_new_terminal_consumer", receipt["mode"])
        self.assertEqual(["node_2"], receipt["eligible_input_agent_ids"])

    def test_unresolved_execution_suppresses_output_closure(self) -> None:
        graph = AgentGraph([_node("node_1"), _node("node_2")])
        env = _env(graph, max_agents=3)
        _seed_artifact(env, "node_1", generated_as_output_agent=False)
        env._unresolved_dirty_agents.add("node_2")

        self.assertFalse(env._requires_new_output_consumer())
        receipt = env._add_output_provenance_domain()
        self.assertNotEqual("required_new_terminal_consumer", receipt["mode"])

    def test_typed_failure_precedes_stale_output_provenance_repair(self) -> None:
        graph = AgentGraph(
            [_node("node_1"), _node("node_2")],
            output_agent_id="node_2",
        )
        env = _env(
            graph,
            recovery_policy="preserve_diagnose_repair_augment",
        )
        _seed_artifact(env, "node_1", generated_as_output_agent=False)
        _seed_artifact(env, "node_2", generated_as_output_agent=False)
        env._record_failure_state(
            (
                AgentFailureRecord(
                    request_id="producer-request",
                    agent_id="node_1",
                    phase=ExecutionPhase.SINGLE,
                    graph_revision=graph.revision,
                    error_type="CompletionArtifactEmpty",
                    message="Agent produced no non-empty completion Artifact",
                    metadata={
                        "public_error_code": "completion_artifact_empty"
                    },
                ),
            ),
            current_agent_ids={"node_1", "node_2"},
        )

        self.assertEqual(("modify_agent",), env.model_admissible_action_types())
        self.assertEqual(
            ["node_1"],
            env.model_admissible_action_targets()["modify_agent"]["agent_ids"],
        )


class HealthBenchOutputClosureAdmissionTests(unittest.IsolatedAsyncioTestCase):
    def test_non_sink_output_closure_ingress_is_rejected_authoritatively(self) -> None:
        graph = AgentGraph(
            [_node("node_1"), _node("node_2"), _node("node_3")],
            [
                AgentRelation("node_1", "node_2", True, False),
                AgentRelation("node_3", "node_2", True, False),
            ],
        )
        env = _env(graph, max_agents=4)
        for agent_id in ("node_1", "node_2", "node_3"):
            _seed_artifact(
                env,
                agent_id,
                generated_as_output_agent=False,
            )

        action = env.parser.parse(
            json.dumps(
                {
                    "action": "add_subgraph",
                    "agents": [
                        {
                            "agent_id": "node_4",
                            "model_id": "model",
                            "contract": (
                                "Consume the routed artifact and produce one "
                                "complete user-facing assistant response."
                            ),
                            "execution_mode": "reasoning",
                            "allowed_tools": [],
                        }
                    ],
                    "relations": [
                        {
                            "source_id": "node_1",
                            "target_id": "node_4",
                            "source_to_target": True,
                            "target_to_source": False,
                        }
                    ],
                    "output_agent_id": "node_4",
                }
            )
        )

        issue = env._add_output_provenance_issue(action)
        self.assertIsNotNone(issue)
        assert issue is not None
        self.assertIn("quotient sink", issue)

    async def test_raw_add_without_same_action_output_is_rejected(self) -> None:
        graph = AgentGraph([_node("node_1")])
        env = _env(graph)
        _seed_artifact(env, "node_1", generated_as_output_agent=False)

        result = await env.step(_add_action(output_agent_id=None))

        self.assertFalse(result.accepted)
        self.assertIn("output_agent_id", result.feedback)
        self.assertIn("same", result.feedback.casefold())
        self.assertEqual("node_1", env.graph.nodes[0].id)
        self.assertEqual(1, len(env.graph.nodes))

    async def test_raw_add_cannot_promote_existing_intermediate_artifact(self) -> None:
        graph = AgentGraph([_node("node_1")])
        env = _env(graph)
        _seed_artifact(env, "node_1", generated_as_output_agent=False)

        result = await env.step(_add_action(output_agent_id="node_1"))

        self.assertFalse(result.accepted)
        self.assertIn("output_agent_id", result.feedback)
        self.assertIn("intermediate", result.feedback.casefold())
        self.assertIn("Output protocol", result.feedback)
        self.assertIsNone(env.graph.output_agent_id)

    async def test_existing_output_protocol_artifact_can_be_selected(self) -> None:
        graph = AgentGraph([_node("node_1")])
        env = _env(graph)
        _seed_artifact(env, "node_1", generated_as_output_agent=True)

        self.assertIn("set_output", env.model_admissible_action_types())
        self.assertEqual(
            ["node_1"],
            env.model_admissible_action_targets()["set_output"]["agent_ids"],
        )
        result = await env.step('{"action":"set_output","agent_id":"node_1"}')
        self.assertTrue(result.accepted, result.feedback)
        self.assertEqual("node_1", env.graph.output_agent_id)

    def test_stale_output_provenance_requires_modifying_current_output(self) -> None:
        graph = AgentGraph([_node("node_1")], output_agent_id="node_1")
        env = _env(graph)
        _seed_artifact(env, "node_1", generated_as_output_agent=False)

        gate = env.finish_admissibility()
        self.assertFalse(gate["admissible"])
        self.assertEqual("output_provenance", gate["stage"])
        self.assertIn("Output execution protocol", gate["reason"])
        self.assertEqual(("modify_agent",), env.model_admissible_action_types())
        self.assertEqual(
            ["node_1"],
            env.model_admissible_action_targets()["modify_agent"]["agent_ids"],
        )


if __name__ == "__main__":
    unittest.main()
