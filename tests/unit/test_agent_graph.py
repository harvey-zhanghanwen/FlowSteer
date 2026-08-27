from __future__ import annotations

import json
import os
import random
import tempfile
from types import SimpleNamespace
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
from src.interactive.agent_runtime import (
    AgentCallRecord,
    AgentFailureRecord,
    AgentRequest,
    AgentResponse,
    AgentRuntimeResult,
    ExecutionPhase,
    UpstreamMessage,
)
from src.interactive.agent_workflow_env import AgentWorkflowEnv, AgentWorkflowStateError
from src.interactive.director import AgentGraphOrchestrator
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


class _ProfileRuntime:
    def __init__(self, model_registry: ModelRegistry) -> None:
        self.model_registry = model_registry

    def registered_execution_profiles(self):
        return (
            ("reasoning", ()),
            ("react", ("qa-retrieval",)),
        )


class _HotpotQAMemoryProfileRuntime:
    def __init__(self, model_registry: ModelRegistry) -> None:
        self.model_registry = model_registry

    def registered_execution_profiles(self):
        return (
            ("reasoning", ()),
            ("react", ("hotpotqa.qa_memory",)),
        )


class EnvironmentTests(unittest.IsolatedAsyncioTestCase):
    async def test_control_plane_feedback_hides_agent_artifact_content(self) -> None:
        registry = make_registry()
        env = AgentWorkflowEnv(
            registry,
            _ImmediateGateway(),
            problem="question",
            execute_on_edit=True,
            director_feedback_mode="control_plane",
        )
        result = await env.step(
            '{"action":"add_agent","agent_id":"worker",'
            '"model_id":"balanced","contract":"produce private evidence"}'
        )

        self.assertTrue(result.accepted)
        self.assertIn('"feedback_mode":"control_plane"', result.feedback)
        self.assertIn('"artifact_id":"revision-1:worker"', result.feedback)
        self.assertNotIn("answer:worker", result.feedback)
        self.assertNotIn("artifact_preview", result.feedback)

        output = await env.step('{"action":"set_output","agent_id":"worker"}')
        self.assertIn('"execution_role":"output"', output.feedback)
        self.assertNotIn('"execution_role":"format"', output.feedback)

    async def test_control_plane_exposes_only_tool_receipt_summary(self) -> None:
        registry = make_registry()
        graph = AgentGraph(
            [AgentNode("worker", "balanced", "retrieve")],
            output_agent_id="worker",
        )
        env = AgentWorkflowEnv(
            registry,
            runtime=_ProfileRuntime(registry),  # type: ignore[arg-type]
            problem="question",
            graph=graph,
            director_feedback_mode="control_plane",
        )
        execution = AgentRuntimeResult(
            run_id="run",
            graph_revision=graph.revision,
            output_agent_id="worker",
            final_answer="<answer>x</answer>",
            outputs={"worker": "private artifact"},
            calls=(),
            block_completion_order=(("worker",),),
            executed_agent_ids=("worker",),
            output_metadata={
                "worker": {
                    "tool_receipts": (
                        {
                            "tool_id": "qa-retrieval",
                            "request": {"action": "search", "query": "private"},
                            "result": {"canonical_answer": "private-answer"},
                            "error_type": None,
                        },
                    )
                }
            },
        )
        feedback = env._accepted_feedback(
            AgentActionParser().parse('{"action":"set_output","agent_id":"worker"}'),
            execution,
        )

        self.assertIn('"tool_receipt_count":1', feedback)
        self.assertIn('"successful_tool_actions":["search"]', feedback)
        self.assertIn('"tool_ids":["qa-retrieval"]', feedback)
        self.assertNotIn("private-answer", feedback)
        self.assertNotIn('"query"', feedback)

    async def test_finish_action_mask_requires_distinct_routed_tool_evidence(self) -> None:
        registry = make_registry()
        singleton = AgentGraph(
            [
                AgentNode(
                    "worker",
                    "balanced",
                    "retrieve and answer",
                    execution_mode="react",
                    allowed_tools=("qa-retrieval",),
                )
            ],
            output_agent_id="worker",
        )
        env = AgentWorkflowEnv(
            registry,
            runtime=_ProfileRuntime(registry),  # type: ignore[arg-type]
            problem="question",
            graph=singleton,
            execute_on_edit=True,
            required_evidence_tool_id="qa-retrieval",
            require_evidence_relation=True,
        )
        receipt = {
            "tool_id": "qa-retrieval",
            "request": {"action": "search"},
            "result": {"hits": ["m1"]},
            "error_type": None,
        }
        read_receipt = {
            "tool_id": "qa-retrieval",
            "request": {"action": "read"},
            "result": {"memory_id": "m1"},
            "error_type": None,
        }
        env._progressive_execution = AgentRuntimeResult(
            run_id="single",
            graph_revision=singleton.revision,
            output_agent_id="worker",
            final_answer="answer",
            outputs={"worker": "answer"},
            calls=(),
            block_completion_order=(("worker",),),
            output_metadata={"worker": {"tool_receipts": (receipt, read_receipt)}},
        )
        env._progressive_execution_revision = singleton.revision
        self.assertNotIn("finish", env.model_admissible_action_types())
        self.assertNotIn(
            "set_output",
            env.model_admissible_action_targets(),
        )

        routed = AgentGraph(
            [
                AgentNode(
                    "worker",
                    "balanced",
                    "retrieve",
                    execution_mode="react",
                    allowed_tools=("qa-retrieval",),
                ),
                AgentNode("output", "balanced", "answer"),
            ],
            [AgentRelation("worker", "output", True, False)],
            output_agent_id="output",
        )
        routed_env = AgentWorkflowEnv(
            registry,
            runtime=_ProfileRuntime(registry),  # type: ignore[arg-type]
            problem="question",
            graph=routed,
            execute_on_edit=True,
            required_evidence_tool_id="qa-retrieval",
            require_evidence_relation=True,
        )
        routed_env._progressive_execution = AgentRuntimeResult(
            run_id="routed",
            graph_revision=routed.revision,
            output_agent_id="output",
            final_answer="answer",
            outputs={"worker": "evidence", "output": "answer"},
            calls=(),
            block_completion_order=(("worker",), ("output",)),
            output_metadata={
                "worker": {"tool_receipts": (receipt, read_receipt)},
                "output": {},
            },
        )
        routed_env._progressive_execution_revision = routed.revision
        self.assertEqual(
            ["output"],
            routed_env.model_admissible_action_targets()["set_output"][
                "agent_ids"
            ],
        )
        self.assertIn("finish", routed_env.model_admissible_action_types())
        self.assertTrue(routed_env.finish_admissibility()["admissible"])

    async def test_hotpotqa_qa_memory_finish_requires_exact_routed_lineage(self) -> None:
        registry = make_registry()
        question = "Who wrote Alpha?"
        artifact = {
            "question_scope": question,
            "memory_id": "memory-1",
            "source_train_task_id": "hotpotqa:train-a",
            "paraphrase_question": "Which person wrote Alpha?",
            "paraphrase_answer_statement": "The writer of Alpha is Ada Lovelace.",
            "canonical_answer": "Ada Lovelace",
        }
        search_receipt = {
            "tool_id": "hotpotqa.qa_memory",
            "request": {
                "action": "search",
                "arguments": {"query": "Alpha author", "k": 2},
            },
            "result": {
                "completed": True,
                "value": {
                    "operation": "search",
                    "memory_ids": ["memory-1"],
                    "hits": [
                        {
                            "memory_id": "memory-1",
                            "source_train_task_id": "hotpotqa:train-a",
                            "paraphrase_question": "Which person wrote Alpha?",
                            "similarity": 0.93,
                            "rank": 1,
                        }
                    ],
                },
            },
            "error_type": None,
        }
        read_receipt = {
            "tool_id": "hotpotqa.qa_memory",
            "request": {
                "action": "read",
                "arguments": {"memory_id": "memory-1"},
            },
            "result": {
                "completed": True,
                "value": {
                    "operation": "read",
                    "memory_id": "memory-1",
                    "memory": {
                        "memory_id": "memory-1",
                        "source_train_task_id": "hotpotqa:train-a",
                        "base_task_id": "hotpotqa:train-a",
                        "cycled": False,
                        "paraphrase_question": "Which person wrote Alpha?",
                        "paraphrase_answer_statement": (
                            "The writer of Alpha is Ada Lovelace."
                        ),
                        "canonical_answer": "Ada Lovelace",
                        "paraphrase_version": "semantic-paraphrase-v1",
                        "paraphrase_provenance": "offline-train-only",
                    },
                },
            },
            "error_type": None,
        }
        receipts = (search_receipt, read_receipt)
        graph = AgentGraph(
            [
                AgentNode(
                    "memory-worker",
                    "balanced",
                    "retrieve grounded evidence",
                    role_family="evidence_retriever",
                    execution_mode="react",
                    allowed_tools=("hotpotqa.qa_memory",),
                ),
                AgentNode(
                    "reasoner",
                    "balanced",
                    "derive a semantic candidate",
                    role_family="reasoner",
                ),
                AgentNode(
                    "verifier",
                    "balanced",
                    "verify the semantic candidate",
                    role_family="verifier",
                ),
                AgentNode(
                    "formatter",
                    "balanced",
                    "format the verified answer",
                    role_family="format",
                ),
            ],
            [
                AgentRelation("memory-worker", "reasoner", True, False),
                AgentRelation("reasoner", "verifier", True, False),
                AgentRelation("verifier", "formatter", True, False),
            ],
            output_agent_id="formatter",
        )
        env = AgentWorkflowEnv(
            registry,
            runtime=_HotpotQAMemoryProfileRuntime(registry),  # type: ignore[arg-type]
            problem=question,
            graph=graph,
            execute_on_edit=True,
            required_evidence_tool_id="hotpotqa.qa_memory",
            require_evidence_relation=True,
        )

        def execution(
            final_answer: str,
            worker_artifact: dict[str, str],
            *,
            omit_receipts_for: str | None = None,
        ) -> AgentRuntimeResult:
            worker_wire = json.dumps(worker_artifact, sort_keys=True)
            stage_specs = (
                ("reasoner", "memory-worker", worker_wire, "Ada Lovelace"),
                ("verifier", "reasoner", "Ada Lovelace", "Ada Lovelace"),
                ("formatter", "verifier", "Ada Lovelace", final_answer),
            )
            calls = []
            for target_id, source_id, content, response in stage_specs:
                request = AgentRequest(
                    request_id=f"qa-memory:{target_id}:single",
                    run_id="qa-memory",
                    graph_revision=graph.revision,
                    problem=question,
                    agent=graph.get_node(target_id),
                    model=ModelSpec("balanced", "fake"),
                    provider=ProviderSpec("fake", kind="test"),
                    phase=ExecutionPhase.SINGLE,
                    is_output_agent=target_id == "formatter",
                    is_format_agent=target_id == "formatter",
                    upstream=(
                        UpstreamMessage(
                            source_id,
                            target_id,
                            content,
                            graph_revision=graph.revision,
                            tool_receipts=(
                                () if omit_receipts_for == target_id else receipts
                            ),
                        ),
                    ),
                )
                calls.append(AgentCallRecord(request, AgentResponse(response)))
            return AgentRuntimeResult(
                run_id="qa-memory",
                graph_revision=graph.revision,
                output_agent_id="formatter",
                final_answer=final_answer,
                outputs={
                    "memory-worker": worker_wire,
                    "reasoner": "Ada Lovelace",
                    "verifier": "Ada Lovelace",
                    "formatter": final_answer,
                },
                calls=tuple(calls),
                block_completion_order=(
                    ("memory-worker",),
                    ("reasoner",),
                    ("verifier",),
                    ("formatter",),
                ),
                output_metadata={
                    "memory-worker": {"tool_receipts": receipts},
                    "reasoner": {"tool_receipts": receipts},
                    "verifier": {"tool_receipts": receipts},
                    "formatter": {},
                },
            )

        env._progressive_execution = execution(
            "<answer>The Ada Lovelace</answer>",
            artifact,
        )
        env._progressive_execution_revision = graph.revision
        self.assertTrue(env.finish_admissibility()["admissible"])
        self.assertEqual(("finish",), env.model_admissible_action_types())

        env._progressive_execution = execution("<answer>Grace Hopper</answer>", artifact)
        mismatch = env.finish_admissibility()
        self.assertFalse(mismatch["admissible"])
        self.assertIn("canonical_answer", str(mismatch["reason"]))

        invalid_artifact = dict(artifact)
        invalid_artifact["memory_id"] = "memory-2"
        env._progressive_execution = execution(
            "<answer>Ada Lovelace</answer>",
            invalid_artifact,
        )
        invalid = env.finish_admissibility()
        self.assertFalse(invalid["admissible"])
        self.assertIn("exact search/read receipt lineage", str(invalid["reason"]))

        env._progressive_execution = execution(
            "<answer>Ada Lovelace</answer>",
            artifact,
            omit_receipts_for="verifier",
        )
        missing_hop = env.finish_admissibility()
        self.assertFalse(missing_hop["admissible"])
        self.assertIn("Verifier", str(missing_hop["reason"]))

    async def test_hotpotqa_qa_memory_finish_rejects_singleton_tool_output(self) -> None:
        registry = make_registry()
        graph = AgentGraph(
            [
                AgentNode(
                    "memory-worker",
                    "balanced",
                    "retrieve and answer",
                    role_family="evidence_retriever",
                    execution_mode="react",
                    allowed_tools=("hotpotqa.qa_memory",),
                )
            ],
            output_agent_id="memory-worker",
        )
        env = AgentWorkflowEnv(
            registry,
            runtime=_HotpotQAMemoryProfileRuntime(registry),  # type: ignore[arg-type]
            problem="Who wrote Alpha?",
            graph=graph,
            execute_on_edit=True,
            require_exact_answer_tag=True,
            required_evidence_tool_id="hotpotqa.qa_memory",
            require_evidence_relation=True,
        )
        env._progressive_execution = AgentRuntimeResult(
            run_id="qa-memory-singleton",
            graph_revision=graph.revision,
            output_agent_id="memory-worker",
            final_answer="<answer>Ada Lovelace</answer>",
            outputs={"memory-worker": "{}"},
            calls=(),
            block_completion_order=(("memory-worker",),),
            output_metadata={"memory-worker": {"tool_receipts": ()}},
        )
        env._progressive_execution_revision = graph.revision

        state = env.finish_admissibility()
        self.assertFalse(state["admissible"])
        self.assertIn("missing role families", str(state["reason"]))

    async def test_hotpotqa_qa_memory_live_roles_and_exhausted_repair_augment(self) -> None:
        registry = make_registry()
        graph = AgentGraph(
            [
                AgentNode(
                    "memory-worker",
                    "balanced",
                    "retrieve evidence",
                    role_family="evidence_retriever",
                    execution_mode="react",
                    allowed_tools=("hotpotqa.qa_memory",),
                )
            ]
        )
        env = AgentWorkflowEnv(
            registry,
            runtime=_HotpotQAMemoryProfileRuntime(registry),  # type: ignore[arg-type]
            problem="Who wrote Alpha?",
            graph=graph,
            required_evidence_tool_id="hotpotqa.qa_memory",
            recovery_policy="preserve_diagnose_repair_augment",
            max_agents=5,
        )

        targets = env.model_admissible_action_targets()
        self.assertNotIn("add_agent", targets)
        add_domain = targets["add_subgraph"]
        self.assertEqual(
            ["reasoner", "verifier", "format"],
            add_domain["admitted_new_role_families"],
        )
        self.assertEqual(
            [{"execution_mode": "reasoning", "allowed_tools": []}],
            add_domain["role_constraints"]["reasoner"]["execution_profiles"],
        )
        self.assertEqual(
            [{
                "execution_mode": "react",
                "allowed_tools": ["hotpotqa.qa_memory"],
            }],
            add_domain["role_constraints"]["evidence_retriever"][
                "execution_profiles"
            ],
        )

        def failure(receipt_count: int) -> AgentFailureRecord:
            return AgentFailureRecord(
                request_id="run:memory-worker:single",
                agent_id="memory-worker",
                phase=ExecutionPhase.SINGLE,
                graph_revision=env.revision,
                error_type="ReactExecutionError",
                message="bounded ReAct worker exhausted",
                metadata={
                    "react_trace": ({"observation_status": "success"},),
                    "tool_receipts": tuple(
                        {"tool_id": "hotpotqa.qa_memory"}
                        for _ in range(receipt_count)
                    ),
                },
            )

        env._record_failure_state((failure(2),))
        self.assertEqual(("modify_agent",), env.model_admissible_action_types())
        repaired = await env.step(
            '{"action":"modify_agent","agent_id":"memory-worker",'
            '"contract":"retry from the preserved Tool receipt prefix"}'
        )
        self.assertTrue(repaired.accepted)
        env._record_failure_state((failure(2),))
        self.assertEqual(
            ("add_subgraph",), env.model_admissible_action_types()
        )
        recovery_domain = env.model_admissible_action_targets()["add_subgraph"]
        self.assertEqual(
            ["evidence_retriever", "repair"],
            recovery_domain["admitted_new_role_families"],
        )

        coverage_env = AgentWorkflowEnv(
            registry,
            runtime=_HotpotQAMemoryProfileRuntime(registry),  # type: ignore[arg-type]
            problem="Who wrote Alpha?",
            graph=graph,
            required_evidence_tool_id="hotpotqa.qa_memory",
            recovery_policy="preserve_diagnose_repair_augment",
            max_agents=5,
        )
        coverage_env._record_failure_state(
            (
                AgentFailureRecord(
                    request_id="run:memory-worker:single",
                    agent_id="memory-worker",
                    phase=ExecutionPhase.SINGLE,
                    graph_revision=coverage_env.revision,
                    error_type="ReactExecutionError",
                    message="bounded ReAct worker exhausted",
                    metadata={
                        "react_trace": ({"observation_status": "terminal_failure"},),
                        "tool_receipts": (
                            {"tool_id": "hotpotqa.qa_memory"},
                        ),
                        "retrieval_failure_type": (
                            "knowledge_base_coverage_failure"
                        ),
                    },
                ),
            )
        )
        self.assertEqual((), coverage_env.model_admissible_action_types())
        self.assertEqual(
            ["memory-worker"],
            coverage_env.recovery_state()[
                "knowledge_base_coverage_failure_agent_ids"
            ],
        )
        diagnosis = AgentGraphOrchestrator(
            registry,
            SimpleNamespace(),  # type: ignore[arg-type]
        ).terminal_canvas_diagnosis(coverage_env)
        self.assertIsNotNone(diagnosis)
        assert diagnosis is not None
        self.assertEqual(
            "canvas_action_domain_exhausted",
            diagnosis["public_error_code"],
        )

    async def test_hotpotqa_relation_domain_exposes_only_missing_semantic_edge(self) -> None:
        registry = make_registry()
        graph = AgentGraph(
            [
                AgentNode(
                    "retriever",
                    "balanced",
                    "retrieve evidence",
                    role_family="evidence_retriever",
                    execution_mode="react",
                    allowed_tools=("hotpotqa.qa_memory",),
                ),
                AgentNode(
                    "reasoner",
                    "balanced",
                    "derive answer",
                    role_family="reasoner",
                ),
                AgentNode(
                    "verifier",
                    "balanced",
                    "verify answer",
                    role_family="verifier",
                ),
                AgentNode(
                    "formatter",
                    "balanced",
                    "format answer",
                    role_family="format",
                ),
            ],
            [
                AgentRelation("retriever", "reasoner", True, False),
                AgentRelation("reasoner", "verifier", True, False),
            ],
        )
        env = AgentWorkflowEnv(
            registry,
            runtime=_HotpotQAMemoryProfileRuntime(registry),  # type: ignore[arg-type]
            problem="Who wrote Alpha?",
            graph=graph,
            required_evidence_tool_id="hotpotqa.qa_memory",
        )

        self.assertEqual(("set_relation",), env.model_admissible_action_types())
        relation_domain = env.model_admissible_action_targets()["set_relation"]
        self.assertEqual(
            [
                {
                    "source_id": "verifier",
                    "target_id": "formatter",
                    "source_to_target": True,
                    "target_to_source": False,
                }
            ],
            relation_domain["candidates"],
        )
        self.assertTrue(relation_domain["endpoints_must_differ"])

        repeated = await env.step(
            '{"action":"set_relation","source_id":"reasoner",'
            '"target_id":"verifier","source_to_target":true,'
            '"target_to_source":false}'
        )
        self.assertFalse(repeated.accepted)
        self.assertIn("exact current", repeated.feedback)

        connected = await env.step(
            '{"action":"set_relation","source_id":"verifier",'
            '"target_id":"formatter","source_to_target":true,'
            '"target_to_source":false}'
        )
        self.assertTrue(connected.accepted)
        self.assertEqual(("set_output",), env.model_admissible_action_types())

    async def test_hotpotqa_modify_domain_preserves_atomic_role_profile(self) -> None:
        registry = make_registry()
        graph = AgentGraph(
            [
                AgentNode(
                    "retriever",
                    "balanced",
                    "retrieve evidence",
                    role_family="evidence_retriever",
                    execution_mode="react",
                    allowed_tools=("hotpotqa.qa_memory",),
                )
            ]
        )
        env = AgentWorkflowEnv(
            registry,
            runtime=_HotpotQAMemoryProfileRuntime(registry),  # type: ignore[arg-type]
            problem="Who wrote Alpha?",
            graph=graph,
            required_evidence_tool_id="hotpotqa.qa_memory",
            recovery_policy="preserve_diagnose_repair_augment",
        )
        env._failed_agent_ids.add("retriever")
        domain = env.model_admissible_action_targets()["modify_agent"]
        candidate = domain["per_agent_candidates"][0]

        self.assertEqual("retriever", candidate["agent_id"])
        self.assertEqual("balanced", candidate["current_values"]["model_id"])
        self.assertNotIn(
            "balanced",
            candidate["discrete_value_domains"]["model_id"],
        )
        self.assertFalse(
            {"role_family", "execution_mode", "allowed_tools"}
            & set(candidate["mutable_fields"])
        )

        half_profile = await env.step(
            '{"action":"modify_agent","agent_id":"retriever",'
            '"execution_mode":"reasoning"}'
        )
        self.assertFalse(half_profile.accepted)
        self.assertIn("per-Agent delta domain", half_profile.feedback)

        current_model = await env.step(
            '{"action":"modify_agent","agent_id":"retriever",'
            '"model_id":"balanced"}'
        )
        self.assertFalse(current_model.accepted)
        self.assertIn("delta domain", current_model.feedback)

    async def test_runtime_profiles_authorize_add_subgraph_and_modify(self) -> None:
        registry = make_registry()
        env = AgentWorkflowEnv(
            registry,
            runtime=_ProfileRuntime(registry),  # type: ignore[arg-type]
            problem="question",
            max_agents=4,
            required_evidence_tool_id="qa-retrieval",
        )

        targets = env.model_admissible_action_targets()
        self.assertEqual(
            [
                {"execution_mode": "reasoning", "allowed_tools": []},
                {
                    "execution_mode": "react",
                    "allowed_tools": ["qa-retrieval"],
                },
            ],
            targets["registered_execution_profiles"],
        )
        added = await env.step(
            '{"action":"add_agent","agent_id":"a","model_id":"balanced",'
            '"contract":"retrieve","execution_mode":"react",'
            '"allowed_tools":["qa-retrieval"]}'
        )
        self.assertTrue(added.accepted)

        coding_modify = await env.step(
            '{"action":"modify_agent","agent_id":"a",'
            '"execution_mode":"coding","allowed_tools":[]}'
        )
        self.assertFalse(coding_modify.accepted)
        self.assertIn("outside the live Runtime capability domain", coding_modify.feedback)

        coding_subgraph = await env.step(
            '{"action":"add_subgraph","agents":[{"agent_id":"b",'
            '"model_id":"balanced","contract":"work",'
            '"execution_mode":"coding","allowed_tools":[]}],"relations":[]}'
        )
        self.assertFalse(coding_subgraph.accepted)
        self.assertEqual(["a"], [node.id for node in env.graph.nodes])

    async def test_delete_requires_typed_unusable_and_complete_takeover(self) -> None:
        registry = make_registry()
        graph = AgentGraph(
            [
                AgentNode(
                    "failed",
                    "balanced",
                    "produce artifact",
                    role_family="research",
                    artifact_type="evidence",
                ),
                AgentNode(
                    "replacement",
                    "balanced",
                    "produce replacement artifact",
                    role_family="research",
                    artifact_type="evidence",
                ),
            ],
            output_agent_id="failed",
        )
        env = AgentWorkflowEnv(
            registry,
            runtime=_ProfileRuntime(registry),  # type: ignore[arg-type]
            problem="question",
            graph=graph,
            recovery_policy="preserve_diagnose_repair_augment",
        )
        env._progressive_outputs["replacement"] = "replacement artifact"

        env._record_failure_state(
            (
                SimpleNamespace(
                    agent_id="failed",
                    metadata={"node_unusable": False},
                ),
            )
        )
        protected = await env.step(
            '{"action":"delete_agent","agent_id":"failed"}'
        )
        self.assertFalse(protected.accepted)

        env._record_failure_state(
            (
                SimpleNamespace(
                    agent_id="failed",
                    metadata={"node_unusable": True},
                ),
            )
        )
        before_takeover = await env.step(
            '{"action":"delete_agent","agent_id":"failed"}'
        )
        self.assertFalse(before_takeover.accepted)
        await env.step('{"action":"set_output","agent_id":"replacement"}')
        deleted = await env.step(
            '{"action":"delete_agent","agent_id":"failed"}'
        )
        self.assertTrue(deleted.accepted)
        self.assertEqual(["replacement"], [node.id for node in env.graph.nodes])

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
        added = await env.step(
            '{"action":"add_agent","agent_id":"a","model_id":"balanced","contract":"answer"}'
        )
        self.assertIsNone(added.execution)
        self.assertIn("execution_error=", added.feedback)

        edited = await env.step('{"action":"set_output","agent_id":"a"}')

        self.assertTrue(edited.accepted)
        self.assertFalse(edited.done)
        self.assertIsNotNone(edited.execution)
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
        # One execution follows each accepted Canvas edit; FINISH reuses the
        # current revision's second result.
        self.assertEqual(2, len(gateway.requests))

    async def test_noop_edit_is_rejected_without_reusing_execution(self) -> None:
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
        self.assertFalse(repeated.accepted)
        self.assertIsNone(repeated.execution)
        self.assertFalse(repeated.execution_reused)
        self.assertFalse(repeated.snapshot.history[-1].execution_reused)
        self.assertIn("made no graph change", repeated.feedback)
        self.assertEqual(2, len(gateway.requests))

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
