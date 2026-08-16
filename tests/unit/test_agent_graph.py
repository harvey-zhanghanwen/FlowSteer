from __future__ import annotations

import os
import random
import tempfile
import unittest

from src.interactive.agent_action_parser import (
    AgentActionParseError,
    AgentActionParser,
    AgentActionType,
)
from src.interactive.agent_graph import (
    AgentGraph,
    AgentNode,
    AgentRelation,
    GraphMutationError,
)
from src.interactive.agent_runtime import AgentRequest
from src.interactive.agent_workflow_env import AgentWorkflowEnv, AgentWorkflowStateError
from src.interactive.model_registry import (
    ModelRegistry,
    ModelRegistryError,
    ModelSpec,
    ProviderSpec,
)


def make_registry() -> ModelRegistry:
    provider = ProviderSpec("fake", kind="test", api_key_env="FAKE_API_KEY")
    return ModelRegistry(
        [provider],
        [
            ModelSpec("balanced", "fake"),
            ModelSpec("cheap", "fake", cheap_weight=9.0, fast_weight=1.0),
            ModelSpec("fast", "fake", cheap_weight=1.0, fast_weight=9.0),
        ],
    )


def codes(graph: AgentGraph, registry: ModelRegistry, complete: bool = True) -> set[str]:
    return {
        issue.code
        for issue in graph.validate(registry, require_complete=complete).issues
    }


class AgentGraphTests(unittest.TestCase):
    def setUp(self) -> None:
        self.registry = make_registry()

    def test_relation_states_reverse_orientation_and_absence(self) -> None:
        graph = AgentGraph([AgentNode("a", "balanced", "A"), AgentNode("b", "fast", "B")])
        self.assertTrue(graph.relation_bits("a", "b").is_independent)

        graph.set_relation("a", "b", True, False)
        forward = graph.relation_bits("a", "b")
        self.assertEqual(
            (True, False),
            (forward.source_to_target, forward.target_to_source),
        )
        reverse = graph.relation_bits("b", "a")
        self.assertEqual((False, True), (reverse.source_to_target, reverse.target_to_source))

        graph.set_relation("a", "b", False, True)
        graph.set_relation("a", "b", True, True)
        self.assertTrue(graph.relation_bits("a", "b").is_bidirectional)
        revision = graph.revision
        graph.set_relation("a", "b", True, True)
        self.assertEqual(revision, graph.revision)
        graph.set_relation("a", "b", False, False)
        self.assertEqual((), graph.relations)

    def test_valid_singleton_chain_fanin_and_bidirectional_output(self) -> None:
        singleton = AgentGraph([AgentNode("a", "balanced", "answer")], output_agent_id="a")
        self.assertTrue(singleton.validate(self.registry).valid)

        chain = AgentGraph(
            [AgentNode("a", "cheap", "draft"), AgentNode("b", "fast", "answer")],
            [AgentRelation("a", "b", True, False)],
            output_agent_id="b",
        )
        self.assertTrue(chain.validate(self.registry).valid)

        fanin = AgentGraph(
            [
                AgentNode("a", "cheap", "left"),
                AgentNode("b", "fast", "right"),
                AgentNode("c", "balanced", "merge"),
            ],
            [AgentRelation("a", "c", True, False), AgentRelation("b", "c", True, False)],
            output_agent_id="c",
        )
        self.assertTrue(fanin.validate(self.registry).valid)

        reciprocal = AgentGraph(
            [AgentNode("a", "cheap", "draft"), AgentNode("b", "fast", "revise")],
            [AgentRelation("a", "b", True, True)],
            output_agent_id="b",
        )
        self.assertTrue(reciprocal.validate(self.registry).valid)

    def test_topology_statistics_report_observed_shape_without_requiring_it(self) -> None:
        graph = AgentGraph(
            [
                AgentNode("a", "cheap", "left evidence"),
                AgentNode("b", "fast", "right evidence"),
                AgentNode("c", "balanced", "synthesize"),
                AgentNode("out", "balanced", "answer"),
            ],
            [
                AgentRelation("a", "c", True, False),
                AgentRelation("b", "c", True, False),
                AgentRelation("c", "out", True, False),
            ],
            output_agent_id="out",
        )

        stats = graph.topology_statistics()

        self.assertEqual(4, stats["agent_count"])
        self.assertEqual(3, stats["directed_edge_count"])
        self.assertEqual(3, stats["max_depth"])
        self.assertEqual(2, stats["max_width"])
        self.assertEqual(["c"], stats["fan_in_agent_ids"])
        self.assertEqual(["a", "b"], stats["root_agent_ids"])

    def test_all_validation_invariants(self) -> None:
        duplicate = AgentGraph(
            [AgentNode("a", "balanced", "one"), AgentNode("a", "balanced", "two")],
            output_agent_id="a",
        )
        self.assertIn("duplicate_agent_id", codes(duplicate, self.registry))

        unknown_and_empty = AgentGraph([AgentNode("a", "missing", "")], output_agent_id="a")
        self.assertTrue({"unknown_model_id", "empty_contract"} <= codes(unknown_and_empty, self.registry))

        self_edge = AgentGraph(
            [AgentNode("a", "balanced", "one")],
            [AgentRelation("a", "a", True, False)],
            output_agent_id="a",
        )
        self.assertIn("self_relation", codes(self_edge, self.registry))

        oversized = AgentGraph(
            [AgentNode(name, "balanced", name) for name in ("a", "b", "c")],
            [AgentRelation("a", "b", True, True), AgentRelation("b", "c", True, True)],
            output_agent_id="c",
        )
        self.assertIn("bidirectional_block_too_large", codes(oversized, self.registry))

        quotient_cycle = AgentGraph(
            [AgentNode(name, "balanced", name) for name in ("a", "b", "c")],
            [
                AgentRelation("a", "b", True, True),
                AgentRelation("b", "c", True, False),
                AgentRelation("c", "a", True, False),
            ],
            output_agent_id="c",
        )
        self.assertIn("quotient_cycle", codes(quotient_cycle, self.registry))

        missing_output = AgentGraph([AgentNode("a", "balanced", "one")])
        self.assertIn("output_agent_count", codes(missing_output, self.registry))
        self.assertNotIn("output_agent_count", codes(missing_output, self.registry, complete=False))

        wrong_output = AgentGraph([AgentNode("a", "balanced", "one")], output_agent_id="missing")
        self.assertIn("unknown_output_agent", codes(wrong_output, self.registry))

        disconnected = AgentGraph(
            [AgentNode("a", "balanced", "one"), AgentNode("b", "balanced", "two")],
            output_agent_id="b",
        )
        self.assertIn("cannot_reach_output", codes(disconnected, self.registry))

        nonsink = AgentGraph(
            [AgentNode("a", "balanced", "one"), AgentNode("b", "balanced", "two")],
            [AgentRelation("a", "b", True, False)],
            output_agent_id="a",
        )
        self.assertIn("output_not_sink", codes(nonsink, self.registry))

    def test_revisions_snapshot_round_trip_and_fork_isolation(self) -> None:
        graph = AgentGraph()
        graph.add_agent(AgentNode("a", "balanced", "one"))
        self.assertEqual(1, graph.revision)
        with self.assertRaises(GraphMutationError):
            graph.add_agent(AgentNode("a", "fast", "duplicate"))
        self.assertEqual(1, graph.revision)
        graph.modify_agent("a", contract="one")
        self.assertEqual(1, graph.revision)
        graph.set_output("a")
        snapshot = graph.snapshot()
        restored = AgentGraph.from_snapshot(snapshot)
        self.assertEqual(snapshot.to_dict(), restored.snapshot().to_dict())
        self.assertEqual(snapshot.snapshot_id, restored.snapshot().snapshot_id)
        fork = graph.fork()
        fork.modify_agent("a", model_id="fast")
        self.assertEqual("balanced", graph.get_node("a").model_id)
        self.assertEqual("fast", fork.get_node("a").model_id)

    def test_delete_cleans_relations_and_output(self) -> None:
        graph = AgentGraph([AgentNode("a", "balanced", "A"), AgentNode("b", "fast", "B")])
        graph.set_relation("a", "b", True, False)
        graph.set_output("b")
        graph.delete_agent("b")
        self.assertEqual((), graph.relations)
        self.assertIsNone(graph.output_agent_id)


class ParserTests(unittest.TestCase):
    def setUp(self) -> None:
        self.parser = AgentActionParser()

    def test_all_six_actions_and_prompt_alias(self) -> None:
        cases = [
            ('{"action":"add_agent","agent_id":"a","model_id":"m","prompt":"draft"}', AgentActionType.ADD_AGENT),
            ('{"action":"modify_agent","agent_id":"a","contract":"verify"}', AgentActionType.MODIFY_AGENT),
            ('{"action":"delete_agent","agent_id":"a"}', AgentActionType.DELETE_AGENT),
            ('{"action":"set_relation","source_id":"a","target_id":"b","source_to_target":true,"target_to_source":false}', AgentActionType.SET_RELATION),
            ('{"action":"set_output","agent_id":"a"}', AgentActionType.SET_OUTPUT),
            ('{"action":"finish"}', AgentActionType.FINISH),
        ]
        for raw, expected in cases:
            with self.subTest(expected=expected):
                self.assertIs(self.parser.parse(raw).action_type, expected)

    def test_first_object_span_and_no_second_action(self) -> None:
        text = 'Reasoning first.\n```json\n{"action":"finish"}\n```\n{"action":"delete_agent","agent_id":"a"}'
        action = self.parser.parse(text)
        self.assertIs(action.action_type, AgentActionType.FINISH)
        self.assertEqual('{"action":"finish"}', text[action.consumed_start:action.consumed_end])

    def test_strict_rejections(self) -> None:
        invalid = [
            '[{"action":"finish"}]',
            '"scalar" {"action":"finish"}',
            '{bad json} {"action":"finish"}',
            '{"action":"finish","action":"finish"}',
            '{"action":"finish","extra":1}',
            '{"action":"set_output","agent_id":null}',
            '{"action":"set_relation","source_id":"a","target_id":"b","source_to_target":1,"target_to_source":false}',
            '{"action":"modify_agent","agent_id":"a"}',
            '{"action":"finish","extra":NaN}',
        ]
        for raw in invalid:
            with self.subTest(raw=raw), self.assertRaises(AgentActionParseError):
                self.parser.parse(raw)


class ModelRegistryTests(unittest.TestCase):
    def test_seeded_selection_is_stable_order_independent_and_rng_local(self) -> None:
        registry = make_registry()
        global_state = random.getstate()
        first = registry.select_weighted(seed=19, candidate_ids=["fast", "cheap"])
        second = registry.select_weighted(seed=19, candidate_ids=["cheap", "fast"])
        self.assertEqual(first, second)
        self.assertEqual(global_state, random.getstate())
        self.assertEqual(
            registry.select_cheap(seed=1).model_id,
            registry.select_cheap(seed=1).model_id,
        )

    def test_immutable_metadata_and_no_inline_secrets(self) -> None:
        registry = ModelRegistry.from_dict(
            {
                "providers": {
                    "fake": {
                        "kind": "test",
                        "api_key_env": "FAKE_API_KEY",
                        "metadata": {"region": "local"},
                    }
                },
                "models": {"m": {"provider_id": "fake", "metadata": {"tier": "dev"}}},
            }
        )
        before = registry.catalog_id
        with self.assertRaises(TypeError):
            registry.require_model("m").metadata["tier"] = "changed"  # type: ignore[index]
        with self.assertRaises(TypeError):
            registry.require_provider("fake").metadata["region"] = "changed"  # type: ignore[index]
        self.assertEqual(before, registry.catalog_id)
        with self.assertRaises(ModelRegistryError):
            ModelRegistry.from_dict(
                {"providers": [{"provider_id": "p", "api_key": "secret"}], "models": []}
            )

    def test_yaml_loader_and_reference_validation(self) -> None:
        payload = """
providers:
  fake:
    kind: test
    api_key_env: FAKE_API_KEY
models:
  m:
    provider_id: fake
    cheap_weight: 2.0
"""
        descriptor, path = tempfile.mkstemp(suffix=".yaml")
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                handle.write(payload)
            registry = ModelRegistry.from_yaml(path)
            self.assertIn("m", registry)
        finally:
            os.unlink(path)
        with self.assertRaises(ModelRegistryError):
            ModelRegistry([ProviderSpec("p")], [ModelSpec("m", "missing")])


class _ImmediateGateway:
    def __init__(self) -> None:
        self.requests: list[AgentRequest] = []

    async def generate(self, request: AgentRequest) -> str:
        self.requests.append(request)
        return f"answer:{request.agent.id}"


class _FailOnceGateway(_ImmediateGateway):
    def __init__(self) -> None:
        super().__init__()
        self._failed = False

    async def generate(self, request: AgentRequest) -> str:
        self.requests.append(request)
        if not self._failed:
            self._failed = True
            raise RuntimeError("temporary executor failure")
        return f"answer:{request.agent.id}"


class _SequenceGateway(_ImmediateGateway):
    def __init__(self, responses: list[str]) -> None:
        super().__init__()
        self.responses = list(responses)

    async def generate(self, request: AgentRequest) -> str:
        self.requests.append(request)
        return self.responses.pop(0)


class EnvironmentTests(unittest.IsolatedAsyncioTestCase):
    async def test_exact_answer_terminal_protocol_rejects_malformed_finish(self) -> None:
        registry = make_registry()
        gateway = _SequenceGateway(["Paris", "<answer>Paris</answer>"])
        env = AgentWorkflowEnv(
            registry,
            gateway,
            problem="question",
            execute_on_edit=True,
            require_exact_answer_tag=True,
        )
        await env.step(
            '{"action":"add_agent","agent_id":"a","model_id":"balanced","contract":"answer"}'
        )
        progressive = await env.step('{"action":"set_output","agent_id":"a"}')
        self.assertIn('"answer_tag_count":0', progressive.feedback)

        rejected = await env.step('{"action":"finish"}')
        self.assertFalse(rejected.accepted)
        self.assertFalse(env.finished)
        self.assertIn("terminal answer must be exactly one", rejected.feedback)
        self.assertEqual(1, len(gateway.requests))

        await env.step(
            '{"action":"modify_agent","agent_id":"a","contract":"answer with exact wrapper"}'
        )
        finished = await env.step('{"action":"finish"}')
        self.assertTrue(finished.accepted)
        self.assertEqual("<answer>Paris</answer>", finished.final_answer)
        self.assertEqual(2, len(gateway.requests))

    async def test_transactional_edits_finish_and_fork(self) -> None:
        registry = make_registry()
        gateway = _ImmediateGateway()
        env = AgentWorkflowEnv(registry, gateway, problem="question")

        rejected = await env.step(
            '{"action":"add_agent","agent_id":"bad","model_id":"missing","contract":"x"}'
        )
        self.assertFalse(rejected.accepted)
        self.assertEqual(0, env.revision)

        added = await env.step(
            '{"action":"add_agent","agent_id":"a","model_id":"balanced","contract":"answer"}'
        )
        self.assertTrue(added.accepted)
        unfinished = await env.step('{"action":"finish"}')
        self.assertFalse(unfinished.accepted)
        self.assertFalse(env.finished)
        await env.step('{"action":"set_output","agent_id":"a"}')

        fork = env.fork()
        await fork.step(
            '{"action":"modify_agent","agent_id":"a","model_id":"fast"}'
        )
        self.assertEqual("balanced", env.graph.get_node("a").model_id)
        self.assertEqual("fast", fork.graph.get_node("a").model_id)

        finished = await env.step('{"action":"finish"}')
        self.assertTrue(finished.accepted)
        self.assertTrue(finished.done)
        self.assertEqual("answer:a", finished.final_answer)
        after = await env.step('{"action":"finish"}')
        self.assertFalse(after.accepted)
        self.assertTrue(after.done)

    async def test_progressive_failure_is_feedback_and_edit_stays_accepted(self) -> None:
        registry = make_registry()
        gateway = _FailOnceGateway()
        env = AgentWorkflowEnv(
            registry,
            gateway,
            problem="question",
            execute_on_edit=True,
        )
        await env.step(
            '{"action":"add_agent","agent_id":"a","model_id":"balanced","contract":"answer"}'
        )

        edited = await env.step('{"action":"set_output","agent_id":"a"}')

        self.assertTrue(edited.accepted)
        self.assertFalse(edited.done)
        self.assertIsNone(edited.execution)
        self.assertIn("execution_error=", edited.feedback)
        self.assertIn("temporary executor failure", edited.feedback)
        self.assertEqual("a", env.graph.output_agent_id)

        retried = await env.step('{"action":"finish"}')
        self.assertTrue(retried.accepted)
        self.assertTrue(retried.done)
        self.assertEqual("answer:a", retried.final_answer)
        self.assertEqual(2, len(gateway.requests))

    async def test_finish_reuses_successful_progressive_execution(self) -> None:
        registry = make_registry()
        gateway = _ImmediateGateway()
        env = AgentWorkflowEnv(
            registry,
            gateway,
            problem="question",
            execute_on_edit=True,
        )
        await env.step(
            '{"action":"add_agent","agent_id":"a","model_id":"balanced","contract":"answer"}'
        )
        progressive = await env.step('{"action":"set_output","agent_id":"a"}')

        finished = await env.step('{"action":"finish"}')

        self.assertTrue(finished.accepted)
        self.assertIs(progressive.execution, finished.execution)
        self.assertTrue(finished.execution_reused)
        self.assertEqual(1, len(gateway.requests))

    async def test_noop_edit_reuses_same_revision_execution(self) -> None:
        registry = make_registry()
        gateway = _ImmediateGateway()
        env = AgentWorkflowEnv(
            registry,
            gateway,
            problem="question",
            execute_on_edit=True,
        )
        await env.step(
            '{"action":"add_agent","agent_id":"a","model_id":"balanced","contract":"answer"}'
        )
        first = await env.step('{"action":"set_output","agent_id":"a"}')
        repeated = await env.step('{"action":"set_output","agent_id":"a"}')

        self.assertEqual(first.revision, repeated.revision)
        self.assertIs(first.execution, repeated.execution)
        self.assertTrue(repeated.execution_reused)
        self.assertTrue(repeated.snapshot.history[-1].execution_reused)
        self.assertEqual(1, len(gateway.requests))

    async def test_history_survives_snapshot_restore_and_fork(self) -> None:
        registry = make_registry()
        env = AgentWorkflowEnv(registry, _ImmediateGateway(), problem="question")
        await env.step("not an action")
        await env.step(
            '{"action":"add_agent","agent_id":"a","model_id":"balanced","contract":"answer"}'
        )
        snapshot = env.snapshot()

        restored = AgentWorkflowEnv(registry, _ImmediateGateway())
        restored.restore(snapshot)
        fork = env.fork(snapshot)

        self.assertEqual(snapshot.history, restored.history)
        self.assertEqual(snapshot.history, fork.history)
        self.assertEqual(2, len(snapshot.history))
        self.assertFalse(snapshot.history[0].accepted)
        self.assertEqual("add_agent", snapshot.history[1].to_dict()["action"]["action"])

    async def test_runtime_agent_limit_rejects_only_new_agents(self) -> None:
        registry = make_registry()
        env = AgentWorkflowEnv(
            registry,
            _ImmediateGateway(),
            problem="question",
            max_agents=1,
        )
        await env.step(
            '{"action":"add_agent","agent_id":"a","model_id":"balanced","contract":"answer"}'
        )
        over_limit = await env.step(
            '{"action":"add_agent","agent_id":"b","model_id":"fast","contract":"check"}'
        )

        self.assertFalse(over_limit.accepted)
        self.assertIn("max_agents=1", over_limit.feedback)
        self.assertEqual(["a"], [node.id for node in env.graph.nodes])

        oversized = AgentGraph(
            [
                AgentNode("a", "balanced", "answer"),
                AgentNode("b", "fast", "check"),
            ]
        )
        with self.assertRaises(AgentWorkflowStateError):
            AgentWorkflowEnv(
                registry,
                _ImmediateGateway(),
                graph=oversized,
                max_agents=1,
            )

    async def test_finish_runtime_failure_is_rejected_and_canvas_can_continue(self) -> None:
        registry = make_registry()
        gateway = _FailOnceGateway()
        env = AgentWorkflowEnv(registry, gateway, problem="question")
        await env.step(
            '{"action":"add_agent","agent_id":"a","model_id":"balanced","contract":"answer"}'
        )
        await env.step('{"action":"set_output","agent_id":"a"}')

        failed = await env.step('{"action":"finish"}')
        self.assertFalse(failed.accepted)
        self.assertFalse(failed.done)
        self.assertFalse(env.finished)
        self.assertIn("execution_error=", failed.feedback)

        changed = await env.step(
            '{"action":"modify_agent","agent_id":"a","contract":"answer concisely"}'
        )
        self.assertTrue(changed.accepted)
        finished = await env.step('{"action":"finish"}')
        self.assertTrue(finished.accepted)
        self.assertEqual("answer:a", finished.final_answer)


if __name__ == "__main__":
    unittest.main()
