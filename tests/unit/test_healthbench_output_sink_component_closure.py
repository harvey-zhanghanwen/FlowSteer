from __future__ import annotations

import json
import unittest
from collections.abc import Iterable, Mapping

from src.interactive.agent_graph import AgentGraph, AgentNode, AgentRelation
from src.interactive.agent_runtime import (
    AgentFailureRecord,
    AgentRuntimeResult,
    ExecutionPhase,
)
from src.interactive.agent_workflow_env import AgentWorkflowEnv
from src.interactive.director import (
    _live_new_agent_ids,
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
        "Produce a useful response artifact from the available conversation.",
    )


def _env(
    graph: AgentGraph,
    *,
    max_agents: int,
    max_relations_per_subgraph: int = 2,
    require_reciprocal_terminal_artifact_lineage: bool = False,
    recovery_policy: str = "default",
) -> AgentWorkflowEnv:
    env = AgentWorkflowEnv(
        _registry(),
        gateway=object(),
        graph=graph,
        max_agents=max_agents,
        max_agents_per_subgraph=3,
        max_relations_per_subgraph=max_relations_per_subgraph,
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
        require_reciprocal_terminal_artifact_lineage=(
            require_reciprocal_terminal_artifact_lineage
        ),
        recovery_policy=recovery_policy,
    )
    env.reset(
        "A healthcare conversation requiring a complete assistant response",
        graph,
    )
    return env


def _seed_artifact(env: AgentWorkflowEnv, agent_id: str) -> None:
    env._progressive_outputs[agent_id] = f"Materialized artifact from {agent_id}"
    env._progressive_output_metadata[agent_id] = {
        "artifact_version": f"{agent_id}:revision-1",
        "generated_as_output_agent": False,
        "input_artifact_versions": {},
    }


def _new_output_declaration(agent_id: str) -> dict[str, object]:
    return {
        "agent_id": agent_id,
        "model_id": "model",
        "contract": (
            "Consume every routed terminal artifact and produce one complete "
            "user-facing assistant response."
        ),
        "execution_mode": "reasoning",
        "allowed_tools": [],
    }


def _raw_closure_add(
    env: AgentWorkflowEnv,
    *,
    output_agent_id: str,
    ingress_agent_ids: Iterable[str],
):
    return env.parser.parse(
        json.dumps(
            {
                "action": "add_subgraph",
                "agents": [_new_output_declaration(output_agent_id)],
                "relations": [
                    {
                        "source_id": source_id,
                        "target_id": output_agent_id,
                        "source_to_target": True,
                        "target_to_source": False,
                    }
                    for source_id in ingress_agent_ids
                ],
                "output_agent_id": output_agent_id,
            }
        )
    )


def _next_live_agent_id(env: AgentWorkflowEnv) -> str:
    domain = env.model_admissible_action_targets()["add_subgraph"]
    existing_agent_ids = domain["existing_agent_ids"]
    assert isinstance(existing_agent_ids, (list, tuple))
    return _live_new_agent_ids(existing_agent_ids, 1)[0]


def _add_schema(env: AgentWorkflowEnv) -> dict[str, object]:
    output_agent_id = _next_live_agent_id(env)
    return json.loads(
        director_live_action_parameter_json_schema_text(
            "add_subgraph",
            env.model_admissible_action_targets(),
            add_agents=[_new_output_declaration(output_agent_id)],
        )
    )


def _relation_candidate_constants(
    schema: Mapping[str, object],
) -> tuple[dict[str, object], ...]:
    properties = schema["properties"]
    assert isinstance(properties, Mapping)
    relations = properties["relations"]
    assert isinstance(relations, Mapping)
    raw_positional_items = relations.get("prefixItems")
    if raw_positional_items is not None:
        assert isinstance(raw_positional_items, list)
        branch_groups = []
        for positional_item in raw_positional_items:
            assert isinstance(positional_item, Mapping)
            branches = positional_item["anyOf"]
            assert isinstance(branches, list)
            branch_groups.append(branches)
    else:
        items = relations["items"]
        assert isinstance(items, Mapping)
        branches = items["anyOf"]
        assert isinstance(branches, list)
        branch_groups = [branches]
    candidates: list[dict[str, object]] = []
    for branches in branch_groups:
        for branch in branches:
            assert isinstance(branch, Mapping)
            branch_properties = branch["properties"]
            assert isinstance(branch_properties, Mapping)
            candidates.append(
                {
                    key: value["const"]
                    for key, value in branch_properties.items()
                    if isinstance(value, Mapping)
                }
            )
    return tuple(candidates)


def _upstream_reciprocal_lineage_graph() -> AgentGraph:
    return AgentGraph(
        [
            _node("peer_a"),
            _node("peer_b"),
            _node("merge"),
            _node("output"),
        ],
        [
            AgentRelation("peer_a", "peer_b", True, True),
            AgentRelation("peer_a", "merge", True, False),
            AgentRelation("merge", "output", True, False),
        ],
        output_agent_id="output",
    )


def _seed_upstream_reciprocal_lineage_state(
    env: AgentWorkflowEnv,
    *,
    include_peer_b_at_merge: bool,
) -> None:
    merge_version = "merge:revision-2" if include_peer_b_at_merge else "merge:revision-1"
    output_version = (
        "output:revision-2" if include_peer_b_at_merge else "output:revision-1"
    )
    outputs = {
        "peer_a": "peer A final revision",
        "peer_b": "peer B final revision",
        "merge": "merged peer revisions",
        "output": "complete assistant response",
    }
    metadata: dict[str, dict[str, object]] = {
        "peer_a": {
            "artifact_version": "peer_a:final-revision",
            "generated_as_output_agent": False,
            "input_artifact_versions": {"peer_b": "peer_b:draft"},
        },
        "peer_b": {
            "artifact_version": "peer_b:final-revision",
            "generated_as_output_agent": False,
            "input_artifact_versions": {"peer_a": "peer_a:draft"},
        },
        "merge": {
            "artifact_version": merge_version,
            "generated_as_output_agent": False,
            "input_artifact_versions": {
                "peer_a": "peer_a:final-revision",
                **(
                    {"peer_b": "peer_b:final-revision"}
                    if include_peer_b_at_merge
                    else {}
                ),
            },
        },
        "output": {
            "artifact_version": output_version,
            "generated_as_output_agent": True,
            "input_artifact_versions": {"merge": merge_version},
        },
    }
    execution = AgentRuntimeResult(
        run_id="upstream-reciprocal-lineage-test",
        graph_revision=env.graph.revision,
        output_agent_id="output",
        final_answer=outputs["output"],
        outputs=outputs,
        calls=(),
        block_completion_order=(("peer_a", "peer_b"), ("merge",), ("output",)),
        output_metadata=metadata,
    )
    env._progressive_outputs = dict(outputs)
    env._progressive_output_metadata = {
        agent_id: dict(item) for agent_id, item in metadata.items()
    }
    env._progressive_execution = execution
    env._progressive_execution_revision = env.graph.revision
    env._unresolved_dirty_agents.clear()


class HealthBenchOutputSinkComponentClosureTests(unittest.TestCase):
    def test_two_independent_sinks_require_one_ingress_per_component(self) -> None:
        graph = AgentGraph([_node("sink_a"), _node("sink_b")])
        env = _env(graph, max_agents=3)
        _seed_artifact(env, "sink_a")
        _seed_artifact(env, "sink_b")

        domain = env.model_admissible_action_targets()["add_subgraph"]
        provenance = domain["output_provenance"]

        self.assertEqual("required_new_terminal_consumer", provenance["mode"])
        self.assertEqual(
            [["sink_a"], ["sink_b"]],
            provenance["required_ingress_component_agent_ids"],
        )
        self.assertEqual(2, provenance["required_ingress_count"])

        output_agent_id = _next_live_agent_id(env)
        schema = _add_schema(env)
        relation_schema = schema["properties"]["relations"]
        self.assertEqual(2, relation_schema["minItems"])
        self.assertEqual(2, relation_schema["maxItems"])
        self.assertEqual(
            {
                ("sink_a", output_agent_id, True, False),
                ("sink_b", output_agent_id, True, False),
            },
            {
                (
                    candidate["source_id"],
                    candidate["target_id"],
                    candidate["source_to_target"],
                    candidate["target_to_source"],
                )
                for candidate in _relation_candidate_constants(schema)
            },
        )

    def test_raw_add_covering_only_one_of_two_sinks_is_rejected(self) -> None:
        graph = AgentGraph([_node("sink_a"), _node("sink_b")])
        env = _env(graph, max_agents=3)
        _seed_artifact(env, "sink_a")
        _seed_artifact(env, "sink_b")

        issue = env._add_output_provenance_issue(
            _raw_closure_add(
                env,
                output_agent_id="output",
                ingress_agent_ids=("sink_a",),
            )
        )

        self.assertIsNotNone(issue)

    def test_reciprocal_sink_is_one_group_and_either_member_is_eligible(self) -> None:
        graph = AgentGraph(
            [_node("peer_a"), _node("peer_b")],
            [AgentRelation("peer_a", "peer_b", True, True)],
        )
        env = _env(graph, max_agents=3)
        _seed_artifact(env, "peer_a")
        _seed_artifact(env, "peer_b")

        provenance = env.model_admissible_action_targets()["add_subgraph"][
            "output_provenance"
        ]
        self.assertEqual(
            [["peer_a", "peer_b"]],
            provenance["required_ingress_component_agent_ids"],
        )
        self.assertEqual(1, provenance["required_ingress_count"])

        schema = _add_schema(env)
        relation_schema = schema["properties"]["relations"]
        self.assertEqual(1, relation_schema["minItems"])
        self.assertEqual(1, relation_schema["maxItems"])
        self.assertEqual(
            {"peer_a", "peer_b"},
            {
                candidate["source_id"]
                for candidate in _relation_candidate_constants(schema)
            },
        )
        for source_id in ("peer_a", "peer_b"):
            with self.subTest(source_id=source_id):
                self.assertIsNone(
                    env._add_output_provenance_issue(
                        _raw_closure_add(
                            env,
                            output_agent_id="output",
                            ingress_agent_ids=(source_id,),
                        )
                    )
                )

    def test_reciprocal_lineage_gate_requires_both_final_revisions(self) -> None:
        graph = AgentGraph(
            [_node("peer_a"), _node("peer_b")],
            [AgentRelation("peer_a", "peer_b", True, True)],
        )
        env = _env(
            graph,
            max_agents=3,
            require_reciprocal_terminal_artifact_lineage=True,
        )
        _seed_artifact(env, "peer_a")
        _seed_artifact(env, "peer_b")

        provenance = env.model_admissible_action_targets()["add_subgraph"][
            "output_provenance"
        ]
        self.assertEqual("required_new_terminal_consumer", provenance["mode"])
        self.assertEqual(
            [["peer_a"], ["peer_b"]],
            provenance["required_ingress_component_agent_ids"],
        )
        self.assertEqual(2, provenance["required_ingress_count"])
        self.assertEqual(
            ["peer_a", "peer_b"],
            provenance["eligible_input_agent_ids"],
        )

        output_agent_id = _next_live_agent_id(env)
        schema = _add_schema(env)
        relation_schema = schema["properties"]["relations"]
        self.assertEqual(2, relation_schema["minItems"])
        self.assertEqual(2, relation_schema["maxItems"])
        self.assertEqual(
            {
                ("peer_a", output_agent_id, True, False),
                ("peer_b", output_agent_id, True, False),
            },
            {
                (
                    candidate["source_id"],
                    candidate["target_id"],
                    candidate["source_to_target"],
                    candidate["target_to_source"],
                )
                for candidate in _relation_candidate_constants(schema)
            },
        )

        for ingress_agent_ids, admitted in (
            (("peer_a",), False),
            (("peer_b",), False),
            (("peer_a", "peer_b"), True),
        ):
            with self.subTest(
                ingress_agent_ids=ingress_agent_ids,
                admitted=admitted,
            ):
                issue = env._add_output_provenance_issue(
                    _raw_closure_add(
                        env,
                        output_agent_id="output",
                        ingress_agent_ids=ingress_agent_ids,
                    )
                )
                if admitted:
                    self.assertIsNone(issue)
                else:
                    self.assertIsNotNone(issue)

    def test_reciprocal_lineage_gate_off_keeps_component_alternatives(self) -> None:
        graph = AgentGraph(
            [_node("peer_a"), _node("peer_b")],
            [AgentRelation("peer_a", "peer_b", True, True)],
        )
        env = _env(
            graph,
            max_agents=3,
            require_reciprocal_terminal_artifact_lineage=False,
        )
        _seed_artifact(env, "peer_a")
        _seed_artifact(env, "peer_b")

        provenance = env.model_admissible_action_targets()["add_subgraph"][
            "output_provenance"
        ]
        self.assertEqual(
            [["peer_a", "peer_b"]],
            provenance["required_ingress_component_agent_ids"],
        )
        self.assertEqual(1, provenance["required_ingress_count"])
        schema = _add_schema(env)
        self.assertEqual(
            {"peer_a", "peer_b"},
            {
                candidate["source_id"]
                for candidate in _relation_candidate_constants(schema)
            },
        )
        for source_id in ("peer_a", "peer_b"):
            with self.subTest(source_id=source_id):
                self.assertIsNone(
                    env._add_output_provenance_issue(
                        _raw_closure_add(
                            env,
                            output_agent_id="output",
                            ingress_agent_ids=(source_id,),
                        )
                    )
                )

    def test_reciprocal_lineage_gate_fails_closed_below_relation_cap(self) -> None:
        graph = AgentGraph(
            [_node("peer_a"), _node("peer_b")],
            [AgentRelation("peer_a", "peer_b", True, True)],
        )
        env = _env(
            graph,
            max_agents=3,
            max_relations_per_subgraph=1,
            require_reciprocal_terminal_artifact_lineage=True,
        )
        _seed_artifact(env, "peer_a")
        _seed_artifact(env, "peer_b")

        action_types = env.model_admissible_action_types()
        targets = env.model_admissible_action_targets()
        provenance = targets.get("add_subgraph", {}).get(
            "output_provenance",
            {},
        )
        self.assertEqual(
            (),
            action_types,
            "an atomic ADD cannot carry both reciprocal final revisions, so "
            "the live domain must terminate as exhausted instead of sampling "
            "an unsatisfiable Output closure",
        )
        self.assertNotEqual(
            "required_new_terminal_consumer",
            provenance.get("mode"),
        )
        self.assertIsNotNone(
            env._add_output_provenance_issue(
                _raw_closure_add(
                    env,
                    output_agent_id="output",
                    ingress_agent_ids=("peer_a",),
                )
            )
        )

    def test_closure_is_suppressed_if_any_sink_lacks_a_successful_artifact(
        self,
    ) -> None:
        def assert_not_required(env: AgentWorkflowEnv) -> None:
            self.assertFalse(env._requires_new_output_consumer())
            provenance = env._add_output_provenance_domain()
            self.assertNotEqual(
                "required_new_terminal_consumer",
                provenance["mode"],
            )

        graph = AgentGraph([_node("sink_a"), _node("sink_b")])

        with self.subTest(state="missing_artifact"):
            env = _env(graph, max_agents=3)
            _seed_artifact(env, "sink_a")
            assert_not_required(env)

        with self.subTest(state="unresolved"):
            env = _env(graph, max_agents=3)
            _seed_artifact(env, "sink_a")
            _seed_artifact(env, "sink_b")
            env._unresolved_dirty_agents.add("sink_b")
            assert_not_required(env)

        with self.subTest(state="failed"):
            env = _env(graph, max_agents=3)
            _seed_artifact(env, "sink_a")
            _seed_artifact(env, "sink_b")
            env._failed_agent_ids.add("sink_b")
            assert_not_required(env)

    def test_relation_cap_does_not_publish_an_unsatisfiable_required_closure(
        self,
    ) -> None:
        graph = AgentGraph(
            [_node("sink_a"), _node("sink_b"), _node("sink_c")]
        )
        env = _env(graph, max_agents=4, max_relations_per_subgraph=2)
        for agent_id in ("sink_a", "sink_b", "sink_c"):
            _seed_artifact(env, agent_id)

        action_types = env.model_admissible_action_types()
        targets = env.model_admissible_action_targets()
        add_domain = targets.get("add_subgraph", {})
        provenance = add_domain.get("output_provenance", {})
        required_closure_published = (
            isinstance(provenance, Mapping)
            and provenance.get("mode")
            == "required_new_terminal_consumer"
        )
        add_is_the_only_forced_action = action_types == ("add_subgraph",)

        self.assertFalse(
            required_closure_published and add_is_the_only_forced_action,
            "the live domain must not force an ADD whose required ingress "
            "count exceeds max_relations_per_subgraph",
        )

    def test_single_sink_keeps_the_existing_atomic_closure_behavior(self) -> None:
        graph = AgentGraph([_node("sink")])
        env = _env(graph, max_agents=2)
        _seed_artifact(env, "sink")

        domain = env.model_admissible_action_targets()["add_subgraph"]
        provenance = domain["output_provenance"]
        self.assertEqual("required_new_terminal_consumer", provenance["mode"])
        self.assertEqual(["sink"], provenance["eligible_input_agent_ids"])
        self.assertEqual(
            [["sink"]],
            provenance["required_ingress_component_agent_ids"],
        )
        self.assertEqual(1, provenance["required_ingress_count"])

        output_agent_id = _next_live_agent_id(env)
        schema = _add_schema(env)
        self.assertIn("output_agent_id", schema["required"])
        self.assertEqual(
            {"const": output_agent_id},
            schema["properties"]["output_agent_id"],
        )
        self.assertIsNone(
            env._add_output_provenance_issue(
                _raw_closure_add(
                    env,
                    output_agent_id="output",
                    ingress_agent_ids=("sink",),
                )
            )
        )


class HealthBenchUpstreamReciprocalLineageRecoveryTests(
    unittest.IsolatedAsyncioTestCase
):
    def _environment(self) -> AgentWorkflowEnv:
        graph = _upstream_reciprocal_lineage_graph()
        env = _env(
            graph,
            max_agents=5,
            max_relations_per_subgraph=2,
            require_reciprocal_terminal_artifact_lineage=True,
        )
        _seed_upstream_reciprocal_lineage_state(
            env,
            include_peer_b_at_merge=False,
        )
        return env

    def test_missing_upstream_peer_exposes_only_acyclic_lineage_relations(
        self,
    ) -> None:
        env = self._environment()

        finish = env.finish_admissibility()
        self.assertFalse(finish["admissible"])
        self.assertEqual("artifact_lineage", finish["stage"])
        self.assertEqual(["peer_b"], finish["unconsumed_agent_ids"])
        self.assertEqual(("set_relation",), env.model_admissible_action_types())

        candidates = env.model_admissible_action_targets()["set_relation"][
            "candidates"
        ]
        self.assertEqual(
            {
                ("peer_b", "merge", True, False),
                ("peer_b", "output", True, False),
            },
            {
                (
                    item["source_id"],
                    item["target_id"],
                    item["source_to_target"],
                    item["target_to_source"],
                )
                for item in candidates
            },
        )
        for item in candidates:
            candidate = env.graph.fork()
            candidate.set_relation(
                item["source_id"],
                item["target_id"],
                item["source_to_target"],
                item["target_to_source"],
            )
            self.assertTrue(
                candidate.validate(_registry(), require_complete=True).valid,
                item,
            )

    async def test_raw_actions_cannot_bypass_upstream_lineage_relation_gate(
        self,
    ) -> None:
        raw_actions = {
            "add_subgraph": (
                '{"action":"add_subgraph","agents":[{'
                '"agent_id":"helper","model_id":"model",'
                '"contract":"produce a distinct helper artifact",'
                '"execution_mode":"reasoning","allowed_tools":[]}],'
                '"relations":[]}'
            ),
            "modify_agent": (
                '{"action":"modify_agent","agent_id":"merge",'
                '"contract":"rewrite the existing merge artifact"}'
            ),
            "finish": '{"action":"finish"}',
            "unrelated_relation": (
                '{"action":"set_relation","source_id":"peer_a",'
                '"target_id":"output","source_to_target":true,'
                '"target_to_source":false}'
            ),
        }
        for action_name, action_text in raw_actions.items():
            with self.subTest(action=action_name):
                env = self._environment()
                revision = env.revision

                rejected = await env.step(action_text)

                self.assertFalse(rejected.accepted, rejected.feedback)
                self.assertEqual(revision, env.revision)
                self.assertIn("peer_b", rejected.feedback)
                self.assertEqual(
                    ("set_relation",),
                    env.model_admissible_action_types(),
                )

    async def test_measured_runtime_repair_preempts_stale_lineage_gate(self) -> None:
        env = _env(
            _upstream_reciprocal_lineage_graph(),
            max_agents=5,
            max_relations_per_subgraph=2,
            require_reciprocal_terminal_artifact_lineage=True,
            recovery_policy="preserve_diagnose_repair_augment",
        )
        _seed_upstream_reciprocal_lineage_state(
            env,
            include_peer_b_at_merge=False,
        )
        env._record_failure_state(
            (
                AgentFailureRecord(
                    request_id="peer-b-react-exhaustion",
                    agent_id="peer_b",
                    phase=ExecutionPhase.SINGLE,
                    graph_revision=env.graph.revision,
                    error_type="ReactExecutionError",
                    message="react agent 'peer_b' exhausted 6 turns",
                    metadata={
                        "react_trace": [
                            {
                                "turn": 6,
                                "observation_status": "schema_invalid",
                                "public_error_code": "state_action_not_admitted",
                            }
                        ]
                    },
                ),
            ),
            current_agent_ids={node.id for node in env.graph.nodes},
        )
        self.assertEqual(("peer_b",), env._mandatory_repair_agent_ids())
        self.assertEqual(("modify_agent",), env.model_admissible_action_types())
        action = env.parser.parse(
            '{"action":"modify_agent","agent_id":"peer_b",'
            '"contract":"Repair the failed execution and return one complete artifact."}'
        )
        self.assertIsNone(env._mandatory_repair_admission_issue(action))
        self.assertIsNotNone(
            env._reciprocal_terminal_lineage_repair_issue(action)
        )

        def unexpected_lineage_gate(_action):
            raise AssertionError(
                "reciprocal lineage gate must wait for measured Runtime repair"
            )

        env._reciprocal_terminal_lineage_repair_issue = unexpected_lineage_gate
        env._informative_contract_admission_issue = (
            lambda _action: "stop after recovery-priority admission"
        )
        result = await env.step(action)

        self.assertFalse(result.accepted)
        self.assertIn("stop after recovery-priority admission", result.feedback)

    async def test_admitted_peer_relation_then_dirty_closure_allows_finish(
        self,
    ) -> None:
        env = self._environment()

        routed = await env.step(
            '{"action":"set_relation","source_id":"peer_b",'
            '"target_id":"merge","source_to_target":true,'
            '"target_to_source":false}'
        )
        self.assertTrue(routed.accepted, routed.feedback)
        self.assertIsNone(routed.execution)
        self.assertNotIn("merge", env._progressive_outputs)
        self.assertNotIn("output", env._progressive_outputs)

        # Simulate the normal Runtime dirty closure: merge consumes both
        # reciprocal final revisions, then the Output consumes that new merge
        # revision. No model/API call is required for this admission test.
        _seed_upstream_reciprocal_lineage_state(
            env,
            include_peer_b_at_merge=True,
        )

        self.assertEqual((), env._unconsumed_reciprocal_terminal_artifact_ids())
        self.assertTrue(env.finish_admissibility()["admissible"])
        self.assertEqual(("finish",), env.model_admissible_action_types())

        finished = await env.step('{"action":"finish"}')
        self.assertTrue(finished.accepted, finished.feedback)
        self.assertTrue(finished.done)
        self.assertTrue(finished.execution_reused)


if __name__ == "__main__":
    unittest.main()
