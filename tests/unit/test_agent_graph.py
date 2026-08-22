from __future__ import annotations

import asyncio
import json
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
    DependencyEdgeEvidence,
    GraphMutationError,
)
from src.interactive.agent_runtime import (
    AgentRequest,
    AgentResponse,
    AgentRuntime,
    ReasoningExecutionAdapter,
)
from src.interactive.agent_workflow_env import (
    AgentWorkflowEnv,
    AgentWorkflowStateError,
    _evidence_span_matches_read,
)
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


def test_evidence_provenance_accepts_typography_but_not_paraphrase() -> None:
    passage = (
        'Margaret "Peggy" Seeger is an American folksinger.  '
        "She married Ewan MacColl."
    )
    assert _evidence_span_matches_read(
        "Margaret 'Peggy' Seeger is an American folksinger. She married Ewan MacColl.",
        passage,
    )
    assert not _evidence_span_matches_read(
        "Peggy Seeger had American nationality and was Ewan MacColl's spouse.",
        passage,
    )


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
        self.assertEqual(3, stats["structural_depth"])
        self.assertEqual(2, stats["max_width"])
        self.assertEqual("fan_in", stats["topology_family"])
        self.assertEqual(["parallel", "fan_in"], stats["topology_motifs"])
        self.assertEqual(["c"], stats["fan_in_agent_ids"])
        self.assertEqual(["a", "b"], stats["root_agent_ids"])

    def test_structural_depth_contracts_reciprocal_pair(self) -> None:
        graph = AgentGraph(
            [AgentNode(name, "balanced", name) for name in ("a", "b", "c", "d")],
            [
                AgentRelation("a", "b", True, True),
                AgentRelation("b", "c", True, False),
                AgentRelation("c", "d", True, False),
            ],
            output_agent_id="d",
        )

        stats = graph.topology_statistics()

        self.assertEqual(3, stats["structural_depth"])
        self.assertEqual(3, stats["component_count"])
        self.assertEqual("serial_3_plus", stats["topology_family"])
        self.assertEqual(["serial_3_plus", "reciprocal"], stats["topology_motifs"])

    def test_dirty_closure_expands_reciprocal_block_and_directed_descendants(self) -> None:
        graph = AgentGraph(
            [AgentNode(name, "balanced", name) for name in ("a", "b", "c", "d")],
            [
                AgentRelation("a", "b", True, True),
                AgentRelation("b", "c", True, False),
                AgentRelation("d", "c", True, False),
            ],
        )

        self.assertEqual({"a", "b", "c"}, graph.dirty_closure({"a"}))
        self.assertEqual({"d", "c"}, graph.dirty_closure({"d"}))
        self.assertEqual(("b", "d"), graph.directed_predecessors("c"))

    def test_construction_progress_is_neutral_atomic_lower_bound(self) -> None:
        empty = AgentGraph()
        self.assertEqual(3, empty.construction_progress()["minimum_remaining_actions"])

        disconnected = AgentGraph(
            [AgentNode(name, "balanced", name) for name in ("a", "b", "c")]
        )
        progress = disconnected.construction_progress()
        self.assertEqual(4, progress["minimum_remaining_actions"])
        self.assertEqual(
            {"add_agent": 0, "set_relation": 2, "set_output": 1, "finish": 1},
            progress["minimum_remaining_breakdown"],
        )

        chain = AgentGraph(
            [AgentNode(name, "balanced", name) for name in ("a", "b", "c")],
            [
                AgentRelation("a", "b", True, False),
                AgentRelation("b", "c", True, False),
            ],
            output_agent_id="c",
        )
        self.assertEqual(1, chain.construction_progress()["minimum_remaining_actions"])

    def test_effective_depth_requires_explicit_graded_evidence(self) -> None:
        graph = AgentGraph(
            [AgentNode(name, "balanced", name) for name in ("a", "b", "c")],
            [
                AgentRelation("a", "b", True, False),
                AgentRelation("b", "c", True, False),
            ],
            output_agent_id="c",
        )

        unverified = graph.effective_dependency_statistics()
        self.assertEqual(3, unverified["structural_depth"])
        self.assertEqual(1, unverified["effective_dependency_depth"])
        self.assertEqual("unverified", unverified["evidence_status"])

        partial = graph.effective_dependency_statistics(
            [DependencyEdgeEvidence("a", "b", "weak", "delivery:a:b")]
        )
        self.assertEqual(2, partial["effective_dependency_depth"])
        self.assertEqual("weak", partial["evidence_status"])
        self.assertEqual(
            "unverified", partial["full_structural_depth_evidence_status"]
        )

        verified = graph.effective_dependency_statistics(
            [
                DependencyEdgeEvidence("a", "b", "verified", "pair:a:b"),
                DependencyEdgeEvidence("b", "c", "verified", "pair:b:c"),
            ]
        )
        self.assertEqual(3, verified["effective_dependency_depth"])
        self.assertEqual(3, verified["verified_dependency_depth"])
        self.assertEqual("verified", verified["evidence_status"])
        with self.assertRaises(ValueError):
            graph.effective_dependency_statistics(
                [DependencyEdgeEvidence("a", "c", "weak")]
            )

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
        graph.add_agent(
            AgentNode("a", "balanced", "one", role_family="  evidence retrieval  ")
        )
        self.assertEqual(1, graph.revision)
        with self.assertRaises(GraphMutationError):
            graph.add_agent(AgentNode("a", "fast", "duplicate"))
        self.assertEqual(1, graph.revision)
        graph.modify_agent("a", contract="one", role_family="evidence retrieval")
        self.assertEqual(1, graph.revision)
        graph.modify_agent("a", role_family="bridge reasoning")
        self.assertEqual(2, graph.revision)
        graph.set_output("a")
        snapshot = graph.snapshot()
        self.assertEqual("bridge reasoning", snapshot.nodes[0].role_family)
        self.assertEqual("bridge reasoning", snapshot.to_dict()["nodes"][0]["role_family"])
        restored = AgentGraph.from_snapshot(snapshot)
        self.assertEqual(snapshot.to_dict(), restored.snapshot().to_dict())
        self.assertEqual(snapshot.snapshot_id, restored.snapshot().snapshot_id)
        fork = graph.fork()
        fork.modify_agent("a", model_id="fast", role_family="format")
        self.assertEqual("balanced", graph.get_node("a").model_id)
        self.assertEqual("fast", fork.get_node("a").model_id)
        self.assertEqual("bridge reasoning", graph.get_node("a").role_family)
        self.assertEqual("format", fork.get_node("a").role_family)

    def test_role_family_is_optional_free_text_metadata(self) -> None:
        legacy = AgentNode("legacy", "balanced", "answer")
        self.assertIsNone(legacy.role_family)
        self.assertNotIn("role_family", legacy.to_dict())

        free_text = AgentNode(
            "analyst",
            "balanced",
            "compare evidence",
            role_family="cross-document comparison",
        )
        self.assertEqual("cross-document comparison", free_text.role_family)
        self.assertEqual(
            "cross-document comparison",
            free_text.to_dict()["role_family"],
        )
        with self.assertRaises(ValueError):
            AgentNode("empty", "balanced", "answer", role_family="  ")
        with self.assertRaises(TypeError):
            AgentNode("typed", "balanced", "answer", role_family=1)  # type: ignore[arg-type]

    def test_has_node_reads_current_canvas_membership(self) -> None:
        graph = AgentGraph([AgentNode("a", "balanced", "answer")])
        self.assertTrue(graph.has_node("a"))
        self.assertFalse(graph.has_node("missing"))
        graph.delete_agent("a")
        self.assertFalse(graph.has_node("a"))

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

    def test_add_and_modify_accept_optional_free_text_role_family(self) -> None:
        added = self.parser.parse(
            '{"action":"add_agent","agent_id":"a","model_id":"m",'
            '"contract":"collect evidence","role_family":" evidence retrieval "}'
        )
        self.assertEqual("evidence retrieval", added.role_family)
        self.assertEqual("evidence retrieval", added.to_dict()["role_family"])

        modified = self.parser.parse(
            '{"action":"modify_agent","agent_id":"a",'
            '"role_family":"cross-document comparison"}'
        )
        self.assertEqual("cross-document comparison", modified.role_family)
        self.assertIsNone(modified.contract)
        self.assertIsNone(modified.model_id)

        legacy = self.parser.parse(
            '{"action":"add_agent","agent_id":"b","model_id":"m",'
            '"contract":"answer"}'
        )
        self.assertIsNone(legacy.role_family)
        self.assertNotIn("role_family", legacy.to_dict())

    def test_first_object_span_and_no_second_action(self) -> None:
        text = 'Reasoning first.\n```json\n{"action":"finish"}\n```\n{"action":"delete_agent","agent_id":"a"}'
        action = self.parser.parse(text)
        self.assertIs(action.action_type, AgentActionType.FINISH)
        self.assertEqual('{"action":"finish"}', text[action.consumed_start:action.consumed_end])

    def test_add_subgraph_accepts_one_to_three_agents_and_consumes_full_object(self) -> None:
        payloads = [
            '{"action":"add_subgraph","agents":['
            '{"agent_id":"a","model_id":"m","contract":"answer"}'
            '],"relations":[]}',
            '{"action":"add_subgraph","agents":['
            '{"agent_id":"a","model_id":"m","prompt":"draft"},'
            '{"agent_id":"b","model_id":"m","contract":"revise"}'
            '],"relations":['
            '{"source_id":"a","target_id":"b",'
            '"source_to_target":true,"target_to_source":true}'
            '],"output_agent_id":"b"}',
            '{"action":"add_subgraph","agents":['
            '{"agent_id":"left","model_id":"m","contract":"left evidence"},'
            '{"agent_id":"right","model_id":"m","contract":"right evidence"},'
            '{"agent_id":"merge","model_id":"m","contract":"synthesize"}'
            '],"relations":['
            '{"source_id":"left","target_id":"merge",'
            '"source_to_target":true,"target_to_source":false},'
            '{"source_id":"right","target_id":"merge",'
            '"source_to_target":true,"target_to_source":false}'
            '],"output_agent_id":"merge"}',
        ]

        for expected_count, payload in enumerate(payloads, start=1):
            with self.subTest(expected_count=expected_count):
                text = f"Canvas analysis.\n{payload}\n" + '{"action":"finish"}'
                action = self.parser.parse(text)
                self.assertIs(action.action_type, AgentActionType.ADD_SUBGRAPH)
                self.assertEqual(expected_count, len(action.agents))
                self.assertEqual(
                    payload,
                    text[action.consumed_start : action.consumed_end],
                )
                self.assertEqual(payload, action.raw_json)

        self.assertIsNone(self.parser.parse(payloads[0]).output_agent_id)
        self.assertNotIn("output_agent_id", self.parser.parse(payloads[0]).to_dict())

    def test_add_subgraph_normalizes_null_output_to_omitted_roundtrip(self) -> None:
        raw = (
            '{"action":"add_subgraph","agents":['
            '{"agent_id":"a","model_id":"m","contract":"collect evidence"}'
            '],"relations":[],"output_agent_id":null}'
        )

        parsed = self.parser.parse(raw)
        self.assertIsNone(parsed.output_agent_id)
        self.assertNotIn("output_agent_id", parsed.to_dict())
        roundtrip = self.parser.parse(json.dumps(parsed.to_dict()))
        self.assertEqual(parsed.to_dict(), roundtrip.to_dict())
        self.assertIsNone(roundtrip.output_agent_id)

        for invalid_output in ('""', '"   "', "1", "false", "[]", "{}"):
            with self.subTest(invalid_output=invalid_output):
                invalid = raw.replace("null", invalid_output)
                with self.assertRaises(AgentActionParseError):
                    self.parser.parse(invalid)

    def test_add_subgraph_rejects_agent_count_and_invalid_relations(self) -> None:
        invalid = [
            '{"action":"add_subgraph","agents":[],"relations":[]}',
            '{"action":"add_subgraph","agents":['
            '{"agent_id":"a","model_id":"m","contract":"a"},'
            '{"agent_id":"b","model_id":"m","contract":"b"},'
            '{"agent_id":"c","model_id":"m","contract":"c"},'
            '{"agent_id":"d","model_id":"m","contract":"d"}'
            '],"relations":[]}',
            '{"action":"add_subgraph","agents":['
            '{"agent_id":"a","model_id":"m","contract":"a"},'
            '{"agent_id":"a","model_id":"m","contract":"duplicate"}'
            '],"relations":[]}',
            '{"action":"add_subgraph","agents":['
            '{"agent_id":"a","model_id":"m","contract":"a"}'
            '],"relations":['
            '{"source_id":"a","target_id":"a",'
            '"source_to_target":true,"target_to_source":false}'
            ']}',
            '{"action":"add_subgraph","agents":['
            '{"agent_id":"a","model_id":"m","contract":"a"},'
            '{"agent_id":"b","model_id":"m","contract":"b"}'
            '],"relations":['
            '{"source_id":"a","target_id":"b",'
            '"source_to_target":false,"target_to_source":false}'
            ']}',
            '{"action":"add_subgraph","agents":['
            '{"agent_id":"a","model_id":"m","contract":"a"},'
            '{"agent_id":"b","model_id":"m","contract":"b"}'
            '],"relations":['
            '{"source_id":"a","target_id":"b",'
            '"source_to_target":true,"target_to_source":false},'
            '{"source_id":"b","target_id":"a",'
            '"source_to_target":false,"target_to_source":true}'
            ']}',
        ]
        for raw in invalid:
            with self.subTest(raw=raw), self.assertRaises(AgentActionParseError):
                self.parser.parse(raw)

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
            '{"action":"add_agent","agent_id":"a","model_id":"m","contract":"x","role_family":""}',
            '{"action":"modify_agent","agent_id":"a","role_family":null}',
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


class _FailAgentGateway(_ImmediateGateway):
    def __init__(self, failed_agent_id: str) -> None:
        super().__init__()
        self.failed_agent_id = failed_agent_id

    async def generate(self, request: AgentRequest) -> str:
        self.requests.append(request)
        if request.agent.id == self.failed_agent_id:
            raise RuntimeError("unusable executor node")
        return f"answer:{request.agent.id}"


class _SequenceGateway(_ImmediateGateway):
    def __init__(self, responses: list[str]) -> None:
        super().__init__()
        self.responses = list(responses)

    async def generate(self, request: AgentRequest) -> str:
        self.requests.append(request)
        return self.responses.pop(0)


class _CountingRuntime(AgentRuntime):
    """Count progressive execution boundaries without changing Executor calls."""

    def __init__(self, model_registry: ModelRegistry, gateway: _ImmediateGateway) -> None:
        super().__init__(model_registry, gateway)
        self.execute_count = 0

    async def execute(self, *args: object, **kwargs: object):  # type: ignore[no-untyped-def, override]
        self.execute_count += 1
        return await super().execute(*args, **kwargs)  # type: ignore[arg-type]


def _hotpot_semantic_graph(*, format_predecessor: str = "verifier") -> AgentGraph:
    return AgentGraph(
        [
            AgentNode(
                "reader",
                "cheap",
                "read database evidence",
                role_family="evidence_retriever",
                artifact_type="retrieval_evidence",
            ),
            AgentNode(
                "reasoner",
                "balanced",
                "align facts to answer slot and select semantic answer",
                role_family="reasoner",
                artifact_type="semantic_candidate",
            ),
            AgentNode(
                "verifier",
                "balanced",
                "verify evidence binding hops and scope without changing candidate",
                role_family="verifier",
                artifact_type="verified_semantic_answer",
            ),
            AgentNode(
                "formatter",
                "fast",
                "wrap the verified candidate without reasoning",
                role_family="format",
                artifact_type="answer_wrapper",
            ),
        ],
        [
            AgentRelation("reader", "reasoner", True, False),
            AgentRelation("reasoner", "verifier", True, False),
            AgentRelation(format_predecessor, "formatter", True, False),
        ],
        output_agent_id="formatter",
    )


class _HotpotSemanticGateway(_ImmediateGateway):
    def __init__(
        self,
        *,
        reasoner_candidate: str = "Paris",
        verifier_candidate: str = "Paris",
        include_read_receipt: bool = True,
        formatter_value: str = "Paris",
        verifier_supported: bool = True,
    ) -> None:
        super().__init__()
        self.reasoner_candidate = reasoner_candidate
        self.verifier_candidate = verifier_candidate
        self.include_read_receipt = include_read_receipt
        self.formatter_value = formatter_value
        self.verifier_supported = verifier_supported

    async def generate(self, request: AgentRequest):  # type: ignore[no-untyped-def]
        self.requests.append(request)
        if request.agent.id == "reader":
            metadata = {}
            if self.include_read_receipt:
                metadata = {
                    "tool_receipts": [
                        {
                            "tool_id": "qa-retrieval",
                            "tool_version": "test-v1",
                            "request": {
                                "action": "read",
                                "arguments": {"passage_id": "p1"},
                            },
                            "result": {
                                "value": {
                                    "operation": "read",
                                    "passage": {
                                        "id": "p1",
                                        "text": (
                                            "Paris is the capital of France. "
                                            "France is a country in Europe."
                                        ),
                                    },
                                },
                                "completed": True,
                            },
                            "error_type": None,
                        }
                    ]
                }
            return AgentResponse("retrieved passage p1", metadata)
        if request.agent.id == "reasoner":
            return json.dumps(
                {
                    "question_scope": request.problem,
                    "answer_slot": {
                        "entity": "France",
                        "relation": "capital",
                        "answer_type": "city",
                        "qualifiers": [],
                        "proposition_index": 0,
                        "answer_field": "object_or_attribute_value",
                    },
                    "evidence_propositions": [
                        {
                            "subject": "France",
                            "relation": "capital",
                            "object_or_attribute_value": self.reasoner_candidate,
                            "qualifiers": [],
                            "evidence_span": "Paris is the capital of France.",
                        },
                        {
                            "subject": "France",
                            "relation": "located in",
                            "object_or_attribute_value": "Europe",
                            "qualifiers": [],
                            "evidence_span": "France is a country in Europe.",
                        },
                    ],
                    "multi_hop_chain": ["France", "capital", "Paris"],
                    "candidate_answer": self.reasoner_candidate,
                    "evidence": ["Paris is the capital of France."],
                }
            )
        if request.agent.id == "verifier":
            return json.dumps(
                {
                    "candidate_answer": self.verifier_candidate,
                    "evidence_supported": self.verifier_supported,
                    "entity_attribute_binding_correct": True,
                    "multi_hop_complete": True,
                    "scope_preserved": True,
                    "verification_status": (
                        "supported" if self.verifier_supported else "unsupported"
                    ),
                }
            )
        if request.agent.id == "formatter":
            return f"<answer>{self.formatter_value}</answer>"
        raise AssertionError(f"unexpected Agent {request.agent.id!r}")


class EnvironmentTests(unittest.IsolatedAsyncioTestCase):
    async def test_execution_contract_is_rejected_before_canvas_commit(self) -> None:
        registry = make_registry()
        gateway = _ImmediateGateway()
        runtime = _CountingRuntime(registry, gateway)
        env = AgentWorkflowEnv(
            registry,
            runtime=runtime,
            problem="question",
            execute_on_edit=True,
        )

        rejected = await env.step(
            '{"action":"add_agent","agent_id":"retriever",'
            '"model_id":"balanced","contract":"retrieve evidence",'
            '"allowed_tools":["qa-retrieval.search"],'
            '"execution_mode":"reasoning"}'
        )

        self.assertFalse(rejected.accepted)
        self.assertIn("execution contract invalid", rejected.feedback)
        self.assertIn("execution_mode='react' or 'coding'", rejected.feedback)
        self.assertEqual(0, env.revision)
        self.assertEqual((), env.graph.nodes)
        self.assertEqual(0, runtime.execute_count)
        self.assertEqual([], gateway.requests)

    async def test_null_output_executes_an_output_free_component(self) -> None:
        registry = make_registry()
        gateway = _ImmediateGateway()
        runtime = _CountingRuntime(registry, gateway)
        env = AgentWorkflowEnv(
            registry,
            runtime=runtime,
            problem="question",
            execute_on_edit=True,
            require_exact_answer_tag=True,
            require_format_agent=True,
        )

        component = await env.step(
            '{"action":"add_subgraph","agents":['
            '{"agent_id":"evidence","model_id":"cheap",'
            '"contract":"collect evidence"},'
            '{"agent_id":"synthesis","model_id":"fast",'
            '"contract":"synthesize evidence"}'
            '],"relations":['
            '{"source_id":"evidence","target_id":"synthesis",'
            '"source_to_target":true,"target_to_source":false}'
            '],"output_agent_id":null}'
        )

        self.assertTrue(component.accepted)
        self.assertIsNotNone(component.action)
        assert component.action is not None
        self.assertIsNone(component.action.output_agent_id)
        self.assertIsNone(env.graph.output_agent_id)
        self.assertIsNone(component.final_answer)
        self.assertEqual(1, runtime.execute_count)
        self.assertEqual(
            {"evidence", "synthesis"},
            set(component.execution.executed_agent_ids),
        )

    async def test_add_subgraph_rolls_back_the_whole_transaction(self) -> None:
        registry = make_registry()
        gateway = _ImmediateGateway()
        runtime = _CountingRuntime(registry, gateway)
        env = AgentWorkflowEnv(
            registry,
            runtime=runtime,
            problem="question",
            execute_on_edit=True,
        )

        rejected = await env.step(
            '{"action":"add_subgraph","agents":['
            '{"agent_id":"a","model_id":"balanced","contract":"draft"},'
            '{"agent_id":"b","model_id":"missing","contract":"verify"}'
            '],"relations":['
            '{"source_id":"a","target_id":"b",'
            '"source_to_target":true,"target_to_source":false}'
            '],"output_agent_id":"b"}'
        )

        self.assertFalse(rejected.accepted)
        self.assertEqual(0, env.revision)
        self.assertEqual((), env.graph.nodes)
        self.assertEqual((), env.graph.relations)
        self.assertIsNone(env.graph.output_agent_id)
        self.assertEqual(0, runtime.execute_count)
        self.assertEqual([], gateway.requests)

    async def test_three_agent_fan_in_subgraph_has_one_runtime_execution_boundary(self) -> None:
        registry = make_registry()
        gateway = _ImmediateGateway()
        runtime = _CountingRuntime(registry, gateway)
        env = AgentWorkflowEnv(
            registry,
            runtime=runtime,
            problem="question",
            execute_on_edit=True,
        )

        added = await env.step(
            '{"action":"add_subgraph","agents":['
            '{"agent_id":"left","model_id":"cheap","contract":"left evidence"},'
            '{"agent_id":"right","model_id":"fast","contract":"right evidence"},'
            '{"agent_id":"merge","model_id":"balanced","contract":"synthesize"}'
            '],"relations":['
            '{"source_id":"left","target_id":"merge",'
            '"source_to_target":true,"target_to_source":false},'
            '{"source_id":"right","target_id":"merge",'
            '"source_to_target":true,"target_to_source":false}'
            '],"output_agent_id":"merge"}'
        )

        self.assertTrue(added.accepted)
        self.assertEqual(1, runtime.execute_count)
        self.assertEqual({"left", "right", "merge"}, set(added.execution.executed_agent_ids))
        self.assertEqual({"left", "right", "merge"}, {item.agent.id for item in gateway.requests})
        self.assertEqual(3, len(gateway.requests))
        self.assertEqual("fan_in", env.graph.topology_statistics()["topology_family"])

        finished = await env.step('{"action":"finish"}')
        self.assertTrue(finished.accepted)
        self.assertTrue(finished.execution_reused)
        self.assertEqual(1, runtime.execute_count)

    async def test_fan_in_component_then_format_uses_two_execution_boundaries(self) -> None:
        registry = make_registry()

        class FormatGateway(_ImmediateGateway):
            async def generate(self, request: AgentRequest) -> str:
                self.requests.append(request)
                if request.is_format_agent:
                    return "<answer>Paris</answer>"
                return f"evidence:{request.agent.id}"

        gateway = FormatGateway()
        runtime = _CountingRuntime(registry, gateway)
        env = AgentWorkflowEnv(
            registry,
            runtime=runtime,
            problem="question",
            execute_on_edit=True,
            require_exact_answer_tag=True,
            require_format_agent=True,
        )

        component = await env.step(
            '{"action":"add_subgraph","agents":['
            '{"agent_id":"left","model_id":"cheap","contract":"left evidence"},'
            '{"agent_id":"right","model_id":"fast","contract":"right evidence"},'
            '{"agent_id":"merge","model_id":"balanced","contract":"join evidence"}'
            '],"relations":['
            '{"source_id":"left","target_id":"merge",'
            '"source_to_target":true,"target_to_source":false},'
            '{"source_id":"right","target_id":"merge",'
            '"source_to_target":true,"target_to_source":false}'
            ']}'
        )
        self.assertTrue(component.accepted)
        self.assertEqual(1, runtime.execute_count)
        self.assertIsNone(env.graph.output_agent_id)
        self.assertIsNone(component.final_answer)
        self.assertEqual(
            {"left", "right", "merge"}, set(component.execution.executed_agent_ids)
        )

        formatted = await env.step(
            '{"action":"add_subgraph","agents":['
            '{"agent_id":"format","model_id":"fast",'
            '"contract":"serialize one answer","role_family":"format"}'
            '],"relations":['
            '{"source_id":"merge","target_id":"format",'
            '"source_to_target":true,"target_to_source":false}'
            '],"output_agent_id":"format"}'
        )
        self.assertTrue(formatted.accepted)
        self.assertEqual(2, runtime.execute_count)
        self.assertEqual(("format",), formatted.execution.executed_agent_ids)
        self.assertEqual(
            {"left", "right", "merge"}, set(formatted.execution.reused_agent_ids)
        )
        self.assertEqual("<answer>Paris</answer>", formatted.final_answer)

        finished = await env.step('{"action":"finish"}')
        self.assertTrue(finished.accepted)
        self.assertTrue(finished.execution_reused)
        self.assertEqual(2, runtime.execute_count)

    async def test_reciprocal_evidence_block_routes_one_answer_to_format(self) -> None:
        registry = make_registry()

        class FormatGateway(_ImmediateGateway):
            async def generate(self, request: AgentRequest) -> str:
                self.requests.append(request)
                if request.is_format_agent:
                    return "<answer>Paris</answer>"
                return f"evidence:{request.agent.id}"

        gateway = FormatGateway()
        runtime = _CountingRuntime(registry, gateway)
        env = AgentWorkflowEnv(
            registry,
            runtime=runtime,
            problem="question",
            execute_on_edit=True,
            require_exact_answer_tag=True,
            require_format_agent=True,
        )

        evidence = await env.step(
            '{"action":"add_subgraph","agents":['
            '{"agent_id":"reader","model_id":"cheap",'
            '"contract":"construct a source-grounded answer"},'
            '{"agent_id":"verifier","model_id":"balanced",'
            '"contract":"independently verify the answer"}'
            '],"relations":['
            '{"source_id":"reader","target_id":"verifier",'
            '"source_to_target":true,"target_to_source":true}'
            ']}'
        )
        formatted = await env.step(
            '{"action":"add_subgraph","agents":['
            '{"agent_id":"format","model_id":"fast",'
            '"contract":"serialize one verified answer","role_family":"format"}'
            '],"relations":['
            '{"source_id":"verifier","target_id":"format",'
            '"source_to_target":true,"target_to_source":false}'
            '],"output_agent_id":"format"}'
        )
        finished = await env.step('{"action":"finish"}')

        self.assertTrue(evidence.accepted)
        self.assertTrue(formatted.accepted)
        self.assertTrue(finished.accepted)
        self.assertEqual(2, runtime.execute_count)
        self.assertIn("reciprocal", env.graph.topology_statistics()["topology_motifs"])
        self.assertIsNone(env.format_agent_issue())
        self.assertEqual(5, len(gateway.requests))
        self.assertEqual(["verifier"], [
            item.source_agent_id
            for item in formatted.execution.calls[-1].request.upstream
        ])
        self.assertEqual("<answer>Paris</answer>", finished.final_answer)

    async def test_two_agent_bidirectional_subgraph_executes_one_bounded_block(self) -> None:
        registry = make_registry()
        gateway = _ImmediateGateway()
        runtime = _CountingRuntime(registry, gateway)
        env = AgentWorkflowEnv(
            registry,
            runtime=runtime,
            problem="question",
            execute_on_edit=True,
        )

        added = await env.step(
            '{"action":"add_subgraph","agents":['
            '{"agent_id":"draft","model_id":"cheap","contract":"propose"},'
            '{"agent_id":"review","model_id":"fast","contract":"review and revise"}'
            '],"relations":['
            '{"source_id":"draft","target_id":"review",'
            '"source_to_target":true,"target_to_source":true}'
            '],"output_agent_id":"review"}'
        )

        self.assertTrue(added.accepted)
        self.assertEqual(1, runtime.execute_count)
        self.assertEqual(("draft", "review"), added.execution.executed_agent_ids)
        self.assertEqual(4, len(added.execution.calls))
        self.assertEqual(
            {"draft", "revision"},
            {call.request.phase.value for call in added.execution.calls},
        )
        self.assertIn("reciprocal", env.graph.topology_statistics()["topology_motifs"])

    async def test_add_subgraph_can_select_a_distinct_format_output_agent(self) -> None:
        registry = make_registry()

        class FormatGateway(_ImmediateGateway):
            async def generate(self, request: AgentRequest) -> str:
                self.requests.append(request)
                if request.is_format_agent:
                    return "<answer>Paris</answer>"
                return "semantic answer: Paris"

        gateway = FormatGateway()
        runtime = _CountingRuntime(registry, gateway)
        env = AgentWorkflowEnv(
            registry,
            runtime=runtime,
            problem="question",
            execute_on_edit=True,
            require_exact_answer_tag=True,
            require_format_agent=True,
        )

        added = await env.step(
            '{"action":"add_subgraph","agents":['
            '{"agent_id":"solver","model_id":"balanced",'
            '"contract":"compute one semantic answer","role_family":"reasoning"},'
            '{"agent_id":"formatter","model_id":"fast",'
            '"contract":"extract the routed answer","role_family":"format"}'
            '],"relations":['
            '{"source_id":"solver","target_id":"formatter",'
            '"source_to_target":true,"target_to_source":false}'
            '],"output_agent_id":"formatter"}'
        )

        self.assertTrue(added.accepted)
        self.assertEqual(1, runtime.execute_count)
        self.assertEqual("formatter", env.graph.output_agent_id)
        self.assertIsNone(env.format_agent_issue())
        output_request = next(
            call.request
            for call in added.execution.calls
            if call.request.agent.id == "formatter"
        )
        self.assertTrue(output_request.is_output_agent)
        self.assertTrue(output_request.is_format_agent)
        self.assertEqual(
            ["solver"],
            [message.source_agent_id for message in output_request.upstream],
        )
        self.assertEqual(
            {
                "admissible": True,
                "graph_revision": env.revision,
                "submission_semantics": "explicit_finish",
            },
            env.finish_admissibility(),
        )

        finished = await env.step('{"action":"finish"}')
        self.assertTrue(finished.accepted)
        self.assertEqual("<answer>Paris</answer>", finished.final_answer)
        self.assertEqual(1, runtime.execute_count)

    async def test_exact_answer_terminal_protocol_rejects_malformed_finish(self) -> None:
        registry = make_registry()
        gateway = _SequenceGateway(["draft", "Paris", "<answer>Paris</answer>"])
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
        self.assertNotIn("answer_tag_count", progressive.feedback)
        self.assertNotIn("exact_single_answer_tag", progressive.feedback)

        rejected = await env.step('{"action":"finish"}')
        self.assertFalse(rejected.accepted)
        self.assertFalse(env.finished)
        self.assertIn("terminal answer must be exactly one", rejected.feedback)
        self.assertIn("answer_tag_count=0", rejected.feedback)
        self.assertEqual(2, len(gateway.requests))

        await env.step(
            '{"action":"modify_agent","agent_id":"a","contract":"answer with exact wrapper"}'
        )
        finished = await env.step('{"action":"finish"}')
        self.assertTrue(finished.accepted)
        self.assertEqual("<answer>Paris</answer>", finished.final_answer)
        self.assertEqual(3, len(gateway.requests))

    async def test_exact_answer_protocol_rejects_multiple_and_nested_wrappers(self) -> None:
        for answer, tag_count in (
            ("<answer>Paris</answer><answer>Lyon</answer>", 2),
            ("<answer><answer>Paris</answer></answer>", 2),
            ("<answer>   </answer>", 1),
        ):
            with self.subTest(answer=answer):
                registry = make_registry()
                gateway = _SequenceGateway(["draft", answer])
                env = AgentWorkflowEnv(
                    registry,
                    gateway,
                    problem="question",
                    execute_on_edit=True,
                    require_exact_answer_tag=True,
                )
                await env.step(
                    '{"action":"add_agent","agent_id":"a","model_id":"balanced",'
                    '"contract":"answer"}'
                )
                progressive = await env.step('{"action":"set_output","agent_id":"a"}')
                self.assertNotIn("answer_tag_count", progressive.feedback)
                self.assertNotIn("exact_single_answer_tag", progressive.feedback)

                rejected = await env.step('{"action":"finish"}')
                self.assertFalse(rejected.accepted)
                self.assertFalse(env.finished)
                self.assertIn(f"answer_tag_count={tag_count}", rejected.feedback)
                if tag_count > 1:
                    self.assertIn("exact_single_answer_tag=False", rejected.feedback)

    async def test_revision_preserving_edit_is_rejected_without_reexecution(self) -> None:
        registry = make_registry()
        gateway = _SequenceGateway(["draft", "not wrapped"])
        env = AgentWorkflowEnv(
            registry,
            gateway,
            problem="question",
            execute_on_edit=True,
            require_exact_answer_tag=True,
        )
        await env.step(
            '{"action":"add_agent","agent_id":"a","model_id":"balanced",'
            '"contract":"answer"}'
        )
        selected = await env.step('{"action":"set_output","agent_id":"a"}')
        self.assertTrue(selected.accepted)
        self.assertEqual(2, len(gateway.requests))

        repeated = await env.step('{"action":"set_output","agent_id":"a"}')
        self.assertFalse(repeated.accepted)
        self.assertIn("action made no graph change", repeated.feedback)
        self.assertEqual(2, len(gateway.requests))

        finish = await env.step('{"action":"finish"}')
        self.assertFalse(finish.accepted)
        self.assertIn("modify the Output Agent contract/model", finish.feedback)

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

        self.assertTrue(added.accepted)
        self.assertIsNone(added.execution)
        self.assertIn("execution_error=", added.feedback)
        self.assertIn("temporary executor failure", added.feedback)

        edited = await env.step('{"action":"set_output","agent_id":"a"}')

        self.assertTrue(edited.accepted)
        self.assertFalse(edited.done)
        self.assertIsNotNone(edited.execution)
        self.assertIn("execution_result=", edited.feedback)
        self.assertEqual("a", env.graph.output_agent_id)

        retried = await env.step('{"action":"finish"}')
        self.assertTrue(retried.accepted)
        self.assertTrue(retried.done)
        self.assertEqual("answer:a", retried.final_answer)
        self.assertEqual(2, len(gateway.requests))

    async def test_failed_dirty_closure_survives_an_unrelated_edit(self) -> None:
        registry = make_registry()

        class ContractGateway(_ImmediateGateway):
            def __init__(self) -> None:
                super().__init__()
                self.failed_revised_a = False

            async def generate(self, request: AgentRequest) -> str:
                self.requests.append(request)
                if (
                    request.agent.id == "a"
                    and request.agent.contract == "left-v2"
                    and not self.failed_revised_a
                ):
                    self.failed_revised_a = True
                    await asyncio.sleep(0.01)
                    raise RuntimeError("revised a failed once")
                return f"{request.agent.id}:{request.agent.contract}"

        gateway = ContractGateway()
        graph = AgentGraph(
            [
                AgentNode("a", "balanced", "left-v1"),
                AgentNode("b", "fast", "right-v1"),
                AgentNode("merge", "balanced", "merge"),
            ],
            [
                AgentRelation("a", "merge", True, False),
                AgentRelation("b", "merge", True, False),
            ],
            output_agent_id="merge",
        )
        env = AgentWorkflowEnv(
            registry,
            gateway,
            graph=graph,
            problem="question",
            execute_on_edit=True,
        )

        initial = await env.step(
            '{"action":"modify_agent","agent_id":"b","contract":"right-v2"}'
        )
        self.assertIsNotNone(initial.execution)

        failed = await env.step(
            '{"action":"modify_agent","agent_id":"a","contract":"left-v2"}'
        )
        self.assertTrue(failed.accepted)
        self.assertIsNone(failed.execution)
        self.assertIsNotNone(failed.partial_execution)
        assert failed.partial_execution is not None
        self.assertEqual({"b": "b:right-v2"}, dict(failed.partial_execution.outputs))
        self.assertEqual(("a", "merge"), env.unresolved_dirty_agent_ids)
        self.assertFalse(env.finish_admissibility()["admissible"])

        repaired = await env.step(
            '{"action":"modify_agent","agent_id":"b","contract":"right-v3"}'
        )
        self.assertIsNotNone(repaired.execution)
        assert repaired.execution is not None
        self.assertEqual(
            ("a", "b", "merge"),
            repaired.execution.executed_agent_ids,
        )
        self.assertEqual((), repaired.execution.reused_agent_ids)
        merge_request = next(
            call.request
            for call in repaired.execution.calls
            if call.request.agent.id == "merge"
        )
        self.assertEqual(
            ["a:left-v2", "b:right-v3"],
            [message.content for message in merge_request.upstream],
        )
        self.assertEqual((), env.unresolved_dirty_agent_ids)

        finished = await env.step('{"action":"finish"}')
        self.assertTrue(finished.accepted)

    async def test_same_run_successful_branch_is_reused_after_sibling_failure(self) -> None:
        registry = make_registry()

        class BranchGateway(_ImmediateGateway):
            def __init__(self) -> None:
                super().__init__()
                self.failed_b = False

            async def generate(self, request: AgentRequest) -> str:
                self.requests.append(request)
                if request.agent.id == "b" and not self.failed_b:
                    self.failed_b = True
                    await asyncio.sleep(0.01)
                    raise RuntimeError("b failed once")
                return f"{request.agent.id}:{request.agent.contract}"

        gateway = BranchGateway()
        env = AgentWorkflowEnv(
            registry,
            gateway,
            problem="question",
            execute_on_edit=True,
        )
        failed = await env.step(
            '{"action":"add_subgraph","agents":['
            '{"agent_id":"a","model_id":"balanced","contract":"left"},'
            '{"agent_id":"b","model_id":"fast","contract":"right"},'
            '{"agent_id":"merge","model_id":"balanced","contract":"merge"}'
            '],"relations":['
            '{"source_id":"a","target_id":"merge",'
            '"source_to_target":true,"target_to_source":false},'
            '{"source_id":"b","target_id":"merge",'
            '"source_to_target":true,"target_to_source":false}'
            '],"output_agent_id":"merge"}'
        )

        self.assertTrue(failed.accepted)
        self.assertIsNotNone(failed.partial_execution)
        assert failed.partial_execution is not None
        self.assertEqual({"a": "a:left"}, dict(failed.partial_execution.outputs))
        self.assertEqual(("b", "merge"), env.unresolved_dirty_agent_ids)

        repaired = await env.step(
            '{"action":"modify_agent","agent_id":"b","contract":"right-v2"}'
        )
        self.assertIsNotNone(repaired.execution)
        assert repaired.execution is not None
        self.assertEqual(("b", "merge"), repaired.execution.executed_agent_ids)
        self.assertEqual(("a",), repaired.execution.reused_agent_ids)
        self.assertEqual(1, len([item for item in gateway.requests if item.agent.id == "a"]))

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
        self.assertEqual(2, len(gateway.requests))

    async def test_finish_requires_the_configured_environment_actor(self) -> None:
        registry = make_registry()
        gateway = _ImmediateGateway()
        env = AgentWorkflowEnv(
            registry,
            gateway,
            problem="complete the interactive task",
            execute_on_edit=True,
            required_tool_id="alfworld.environment",
        )
        await env.step(
            '{"action":"add_subgraph","agents":['
            '{"agent_id":"solver","model_id":"balanced",'
            '"contract":"solve without the environment"}'
            '],"relations":[],"output_agent_id":"solver"}'
        )

        rejected = await env.step('{"action":"finish"}')

        self.assertFalse(rejected.accepted)
        self.assertFalse(env.finished)
        self.assertIn("exactly one ReAct environment actor", rejected.feedback)
        self.assertIn("alfworld.environment", rejected.feedback)

    async def test_canvas_rejects_actions_outside_configured_search_space(self) -> None:
        registry = make_registry()
        env = AgentWorkflowEnv(
            registry,
            _ImmediateGateway(),
            problem="question",
            execute_on_edit=True,
            allowed_actions=(
                "add_subgraph",
                "modify_agent",
                "delete_agent",
                "set_relation",
                "set_output",
                "finish",
            ),
        )

        rejected = await env.step(
            '{"action":"add_agent","agent_id":"a",'
            '"model_id":"balanced","contract":"answer"}'
        )

        self.assertFalse(rejected.accepted)
        self.assertEqual((), env.graph.nodes)
        self.assertIn("configured Canvas action set", rejected.feedback)

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
        self.assertIn("action made no graph change", repeated.feedback)
        self.assertEqual(2, len(gateway.requests))

    async def test_each_edit_executes_only_dirty_topological_blocks(self) -> None:
        registry = make_registry()
        gateway = _ImmediateGateway()
        env = AgentWorkflowEnv(
            registry,
            gateway,
            problem="question",
            execute_on_edit=True,
        )

        first = await env.step(
            '{"action":"add_agent","agent_id":"a","model_id":"balanced","contract":"evidence"}'
        )
        second = await env.step(
            '{"action":"add_agent","agent_id":"b","model_id":"fast","contract":"consume a"}'
        )
        related = await env.step(
            '{"action":"set_relation","source_id":"a","target_id":"b",'
            '"source_to_target":true,"target_to_source":false}'
        )

        self.assertEqual(("a",), first.execution.executed_agent_ids)
        self.assertEqual(("b",), second.execution.executed_agent_ids)
        self.assertEqual(("a",), second.execution.reused_agent_ids)
        self.assertEqual(("b",), related.execution.executed_agent_ids)
        self.assertEqual(("a",), related.execution.reused_agent_ids)
        self.assertEqual(["a", "b", "b"], [item.agent.id for item in gateway.requests])

    async def test_reciprocal_edit_executes_one_bounded_two_agent_block(self) -> None:
        registry = make_registry()
        gateway = _ImmediateGateway()
        env = AgentWorkflowEnv(
            registry,
            gateway,
            problem="question",
            execute_on_edit=True,
        )
        await env.step(
            '{"action":"add_agent","agent_id":"a","model_id":"balanced","contract":"proposal"}'
        )
        await env.step(
            '{"action":"add_agent","agent_id":"b","model_id":"fast","contract":"peer review"}'
        )
        reciprocal = await env.step(
            '{"action":"set_relation","source_id":"a","target_id":"b",'
            '"source_to_target":true,"target_to_source":true}'
        )

        self.assertEqual(("a", "b"), reciprocal.execution.executed_agent_ids)
        block_calls = reciprocal.execution.calls
        self.assertEqual(4, len(block_calls))
        self.assertEqual(
            {"draft", "revision"},
            {call.request.phase.value for call in block_calls},
        )

    async def test_format_agent_is_terminal_singleton_with_one_semantic_input(self) -> None:
        registry = make_registry()

        class FormatGateway(_ImmediateGateway):
            async def generate(self, request: AgentRequest) -> str:
                self.requests.append(request)
                if request.is_format_agent:
                    return "<answer>Paris</answer>"
                return "semantic answer: Paris"

        gateway = FormatGateway()
        env = AgentWorkflowEnv(
            registry,
            gateway,
            problem="question",
            execute_on_edit=True,
            require_exact_answer_tag=True,
            require_format_agent=True,
        )
        await env.step(
            '{"action":"add_agent","agent_id":"solver","model_id":"balanced","contract":"compute semantic answer"}'
        )
        await env.step(
            '{"action":"add_agent","agent_id":"formatter","model_id":"fast",'
            '"contract":"extract upstream answer only","role_family":"format"}'
        )
        self.assertEqual("format", env.graph.get_node("formatter").role_family)
        premature = await env.step(
            '{"action":"set_output","agent_id":"formatter"}'
        )
        self.assertFalse(premature.accepted)
        self.assertIn("exactly one upstream semantic-answer artifact", premature.feedback)
        self.assertIsNone(env.graph.output_agent_id)
        await env.step(
            '{"action":"set_relation","source_id":"solver","target_id":"formatter",'
            '"source_to_target":true,"target_to_source":false}'
        )
        selected = await env.step(
            '{"action":"set_output","agent_id":"formatter"}'
        )
        finished = await env.step('{"action":"finish"}')

        format_request = selected.execution.calls[-1].request
        self.assertTrue(format_request.is_output_agent)
        self.assertTrue(format_request.is_format_agent)
        self.assertEqual(["solver"], [item.source_agent_id for item in format_request.upstream])
        self.assertIsNone(env.format_agent_issue())
        self.assertTrue(finished.accepted)
        self.assertEqual("<answer>Paris</answer>", finished.final_answer)

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

    async def test_hotpot_semantic_protocol_accepts_only_verified_answer_lineage(self) -> None:
        registry = make_registry()
        gateway = _HotpotSemanticGateway()
        env = AgentWorkflowEnv(
            registry,
            gateway,
            graph=_hotpot_semantic_graph(),
            problem="What is the capital of France?",
            require_exact_answer_tag=True,
            require_format_agent=True,
            semantic_protocol="hotpotqa_verified_answer_slot_v1",
            recovery_policy="preserve_diagnose_repair_augment",
            required_evidence_tool_id="qa-retrieval",
        )

        self.assertFalse(env.finish_admissibility()["admissible"])
        finished = await env.step('{"action":"finish"}')

        self.assertTrue(finished.accepted)
        self.assertEqual("<answer>Paris</answer>", finished.final_answer)
        self.assertEqual(
            {
                "admissible": True,
                "graph_revision": env.revision,
                "submission_semantics": "explicit_finish",
            },
            env.finish_admissibility(),
        )

    async def test_hotpot_finish_and_admissibility_share_candidate_gate(self) -> None:
        registry = make_registry()
        gateway = _HotpotSemanticGateway(verifier_candidate="Lyon")
        env = AgentWorkflowEnv(
            registry,
            gateway,
            graph=_hotpot_semantic_graph(),
            problem="What is the capital of France?",
            require_exact_answer_tag=True,
            require_format_agent=True,
            semantic_protocol="hotpotqa_verified_answer_slot_v1",
            recovery_policy="preserve_diagnose_repair_augment",
            required_evidence_tool_id="qa-retrieval",
        )

        rejected = await env.step('{"action":"finish"}')

        self.assertFalse(rejected.accepted)
        self.assertIn("Verifier changed the Reasoner's candidate_answer", rejected.feedback)
        self.assertFalse(env.finish_admissibility()["admissible"])
        self.assertIsNotNone(rejected.execution)

    async def test_hotpot_semantic_gate_requires_direct_successful_read_receipt(self) -> None:
        registry = make_registry()
        gateway = _HotpotSemanticGateway(include_read_receipt=False)
        env = AgentWorkflowEnv(
            registry,
            gateway,
            graph=_hotpot_semantic_graph(),
            problem="What is the capital of France?",
            require_exact_answer_tag=True,
            require_format_agent=True,
            semantic_protocol="hotpotqa_verified_answer_slot_v1",
            recovery_policy="preserve_diagnose_repair_augment",
            required_evidence_tool_id="qa-retrieval",
        )

        rejected = await env.step('{"action":"finish"}')

        self.assertFalse(rejected.accepted)
        self.assertIn("successful 'qa-retrieval' read receipt", rejected.feedback)
        self.assertFalse(env.finish_admissibility()["admissible"])

    async def test_hotpot_retrieval_evidence_must_route_through_reasoner(self) -> None:
        graph = _hotpot_semantic_graph()
        graph.set_relation("reader", "reasoner", False, False)
        graph.set_relation("reader", "verifier", True, False)
        env = AgentWorkflowEnv(
            make_registry(),
            _HotpotSemanticGateway(),
            graph=graph,
            problem="What is the capital of France?",
            require_exact_answer_tag=True,
            require_format_agent=True,
            semantic_protocol="hotpotqa_verified_answer_slot_v1",
            recovery_policy="preserve_diagnose_repair_augment",
            required_evidence_tool_id="qa-retrieval",
        )

        rejected = await env.step('{"action":"finish"}')

        self.assertFalse(rejected.accepted)
        self.assertIn("Route any retrieval or repair evidence into the Reasoner", rejected.feedback)

    async def test_hotpot_formatter_must_wrap_exact_unchanged_candidate(self) -> None:
        registry = make_registry()
        gateway = _HotpotSemanticGateway(formatter_value="Paris, France")
        env = AgentWorkflowEnv(
            registry,
            gateway,
            graph=_hotpot_semantic_graph(),
            problem="What is the capital of France?",
            require_exact_answer_tag=True,
            require_format_agent=True,
            semantic_protocol="hotpotqa_verified_answer_slot_v1",
            recovery_policy="preserve_diagnose_repair_augment",
            required_evidence_tool_id="qa-retrieval",
        )

        rejected = await env.step('{"action":"finish"}')

        self.assertFalse(rejected.accepted)
        self.assertIn("Formatter must only wrap", rejected.feedback)
        self.assertIn("wrapper_content='Paris, France'", rejected.feedback)
        self.assertFalse(env.finish_admissibility()["admissible"])

    async def test_hotpot_structured_gate_rejects_legacy_untyped_answer_slot(self) -> None:
        registry = make_registry()
        env = AgentWorkflowEnv(
            registry,
            _ImmediateGateway(),
            problem="question",
            semantic_protocol="hotpotqa_verified_answer_slot_v1",
            recovery_policy="preserve_diagnose_repair_augment",
            required_evidence_tool_id="qa-retrieval",
        )
        reasoner_candidate, reasoner_issue = env._reasoner_candidate(
            "Question scope: the capital relation exactly as asked\n"
            "Answer slot: city\n"
            "Evidence propositions: [\"capital(France, Paris)\"]\n"
            "Multi-hop chain: [\"France\", \"Paris\"]\n"
            "Candidate answer: Paris\n"
            "Evidence: Paris is the capital of France."
        )
        verifier_candidate, verifier_issue = env._verifier_candidate(
            "Candidate answer: Paris\n"
            "Evidence supported: true\n"
            "Entity attribute binding correct: true\n"
            "Multi-hop complete: true\n"
            "Scope preserved: true\n"
            "Verification status: supported"
        )

        self.assertIsNone(reasoner_candidate)
        self.assertIn("answer_slot", reasoner_issue or "")
        self.assertIsNone(verifier_issue)
        self.assertEqual("Paris", verifier_candidate)

    async def test_hotpot_structured_answer_slot_binds_candidate_and_exact_scope(
        self,
    ) -> None:
        env = AgentWorkflowEnv(
            make_registry(),
            _ImmediateGateway(),
            problem="What is the capital of France?",
            semantic_protocol="hotpotqa_verified_answer_slot_v1",
            recovery_policy="preserve_diagnose_repair_augment",
            required_evidence_tool_id="qa-retrieval",
        )
        artifact = {
            "question_scope": "What is the capital of France?",
            "answer_slot": {
                "entity": "France",
                "relation": "capital",
                "answer_type": "city",
                "qualifiers": [],
                "proposition_index": 0,
                "answer_field": "object_or_attribute_value",
            },
            "evidence_propositions": [
                {
                    "subject": "France",
                    "relation": "capital",
                    "object_or_attribute_value": "Paris",
                    "qualifiers": [],
                    "evidence_span": "Paris is the capital of France.",
                },
                {
                    "subject": "France",
                    "relation": "located in",
                    "object_or_attribute_value": "Europe",
                    "qualifiers": [],
                    "evidence_span": "France is a country in Europe.",
                },
            ],
            "multi_hop_chain": ["France", "capital", "Paris"],
            "candidate_answer": "Paris",
            "evidence": ["Paris is the capital of France."],
        }

        candidate, issue = env._reasoner_candidate(
            json.dumps(artifact),
            original_question="What is the capital of France?",
        )
        self.assertEqual("Paris", candidate)
        self.assertIsNone(issue)

        narrowed = dict(artifact)
        narrowed["question_scope"] = "What is the singles capital of France?"
        candidate, issue = env._reasoner_candidate(
            json.dumps(narrowed),
            original_question="What is the capital of France?",
        )
        self.assertIsNone(candidate)
        self.assertIn("copy the original question exactly", str(issue))

        rebound = dict(artifact)
        rebound["candidate_answer"] = "Lyon"
        candidate, issue = env._reasoner_candidate(
            json.dumps(rebound),
            original_question="What is the capital of France?",
        )
        self.assertIsNone(candidate)
        self.assertIn("copy answer_slot.answer_field", str(issue))

        coreferential = json.loads(json.dumps(artifact))
        coreferential["evidence_propositions"][0]["evidence_span"] = (
            "It has Paris as its capital."
        )
        candidate, issue = env._reasoner_candidate(
            json.dumps(coreferential),
            original_question="What is the capital of France?",
        )
        self.assertEqual("Paris", candidate)
        self.assertIsNone(issue)

    async def test_hotpot_rejects_react_role_but_not_react_execution_semantics(self) -> None:
        registry = make_registry()
        env = AgentWorkflowEnv(
            registry,
            _ImmediateGateway(),
            problem="question",
            semantic_protocol="hotpotqa_verified_answer_slot_v1",
            recovery_policy="preserve_diagnose_repair_augment",
            required_evidence_tool_id="qa-retrieval",
        )

        rejected = await env.step(
            '{"action":"add_agent","agent_id":"bad","model_id":"balanced",'
            '"contract":"retrieve","role_family":"ReAct"}'
        )

        self.assertFalse(rejected.accepted)
        self.assertEqual((), env.graph.nodes)
        self.assertIn("ReAct is an execution_mode", rejected.feedback)

    async def test_hotpot_format_predecessor_must_be_verifier(self) -> None:
        registry = make_registry()
        gateway = _ImmediateGateway()
        graph = AgentGraph(
            [
                AgentNode("reader", "cheap", "evidence", role_family="evidence_retriever"),
                AgentNode("verifier", "balanced", "verify", role_family="verifier"),
                AgentNode("reasoner", "balanced", "answer", role_family="reasoner"),
                AgentNode("formatter", "fast", "format", role_family="format"),
            ],
            [
                AgentRelation("reader", "verifier", True, False),
                AgentRelation("verifier", "reasoner", True, False),
                AgentRelation("reasoner", "formatter", True, False),
            ],
            output_agent_id="formatter",
        )
        env = AgentWorkflowEnv(
            registry,
            gateway,
            graph=graph,
            problem="question",
            require_exact_answer_tag=True,
            require_format_agent=True,
            semantic_protocol="hotpotqa_verified_answer_slot_v1",
            recovery_policy="preserve_diagnose_repair_augment",
            required_evidence_tool_id="qa-retrieval",
        )

        rejected = await env.step('{"action":"finish"}')

        self.assertFalse(rejected.accepted)
        self.assertIn("unique predecessor must have role_family='verifier'", rejected.feedback)
        self.assertEqual([], gateway.requests)

    async def test_hotpot_format_execution_contract_is_rejected_before_commit(
        self,
    ) -> None:
        registry = make_registry()
        gateway = _ImmediateGateway()
        runtime = AgentRuntime(
            registry,
            gateway,
            execution_adapters={"react": ReasoningExecutionAdapter(gateway)},
            semantic_protocol="hotpotqa_verified_answer_slot_v1",
        )
        env = AgentWorkflowEnv(
            registry,
            runtime=runtime,
            problem="question",
            require_exact_answer_tag=True,
            require_format_agent=True,
            semantic_protocol="hotpotqa_verified_answer_slot_v1",
            recovery_policy="preserve_diagnose_repair_augment",
            required_evidence_tool_id="qa-retrieval",
        )

        rejected = await env.step(
            '{"action":"add_subgraph","agents":['
            '{"agent_id":"reasoner","model_id":"balanced",'
            '"contract":"answer","role_family":"reasoner"},'
            '{"agent_id":"verifier","model_id":"balanced",'
            '"contract":"verify","role_family":"verifier"},'
            '{"agent_id":"formatter","model_id":"fast",'
            '"contract":"format","role_family":"format",'
            '"execution_mode":"react"}'
            '],"relations":['
            '{"source_id":"reasoner","target_id":"verifier",'
            '"source_to_target":true,"target_to_source":false},'
            '{"source_id":"verifier","target_id":"formatter",'
            '"source_to_target":true,"target_to_source":false}'
            '],"output_agent_id":"formatter"}'
        )

        self.assertFalse(rejected.accepted)
        self.assertIn("Format Agent must use reasoning execution", rejected.feedback)
        self.assertEqual((), env.graph.nodes)

    async def test_preserve_repair_policy_blocks_delete_until_takeover(self) -> None:
        registry = make_registry()
        gateway = _FailAgentGateway("source")
        graph = AgentGraph(
            [
                AgentNode(
                    "source",
                    "balanced",
                    "evidence-v1",
                    role_family="evidence",
                    artifact_type="facts",
                ),
                AgentNode("out", "fast", "answer", role_family="output"),
            ],
            [AgentRelation("source", "out", True, False)],
            output_agent_id="out",
        )
        env = AgentWorkflowEnv(
            registry,
            gateway,
            graph=graph,
            problem="question",
            execute_on_edit=True,
            recovery_policy="preserve_diagnose_repair_augment",
        )
        executed = await env.step(
            '{"action":"modify_agent","agent_id":"source",'
            '"contract":"evidence-v2"}'
        )
        self.assertTrue(executed.accepted)
        self.assertIn("execution_error", executed.feedback)
        self.assertIn('"recovery_state"', executed.feedback)
        recovery_state = env.recovery_state()
        self.assertEqual(["source"], recovery_state["failed_agent_ids"])
        self.assertIn(
            "not_diagnosed_unusable",
            recovery_state["deletion_protected"]["out"],
        )

        protected = await env.step('{"action":"delete_agent","agent_id":"source"}')
        self.assertFalse(protected.accepted)
        self.assertIn("preserve_diagnose_repair_augment protects", protected.feedback)
        self.assertIn("modify_agent, set_relation, or add_subgraph", protected.feedback)
        self.assertTrue(env.graph.has_node("source"))

        takeover = await env.step(
            '{"action":"add_subgraph","agents":['
            '{"agent_id":"replacement","model_id":"cheap",'
            '"contract":"replacement evidence","role_family":"evidence",'
            '"artifact_type":"facts"}'
            '],"relations":['
            '{"source_id":"replacement","target_id":"out",'
            '"source_to_target":true,"target_to_source":false}'
            ']}'
        )
        self.assertTrue(takeover.accepted)
        self.assertIn("replacement", env.recovery_state()["preserved_agent_ids"])

        deleted = await env.step('{"action":"delete_agent","agent_id":"source"}')
        self.assertTrue(deleted.accepted)
        self.assertFalse(env.graph.has_node("source"))
        self.assertTrue(env.graph.has_node("replacement"))

    async def test_hotpot_reasoner_takeover_requires_valid_semantic_artifact(self) -> None:
        graph = _hotpot_semantic_graph()
        graph.add_agent(
            AgentNode(
                "replacement_reasoner",
                "cheap",
                "replacement semantic alignment",
                role_family="reasoner",
                artifact_type="semantic_candidate",
            )
        )
        graph.set_relation("replacement_reasoner", "verifier", True, False)
        env = AgentWorkflowEnv(
            make_registry(),
            _ImmediateGateway(),
            graph=graph,
            problem="What is the capital of France?",
            semantic_protocol="hotpotqa_verified_answer_slot_v1",
            recovery_policy="preserve_diagnose_repair_augment",
            required_evidence_tool_id="qa-retrieval",
        )
        env._failed_agent_ids.add("reasoner")
        env._unresolved_dirty_agents.add("reasoner")
        env._progressive_outputs["replacement_reasoner"] = "Paris"
        env._progressive_output_metadata["replacement_reasoner"] = {}

        invalid_issue = env._delete_admission_issue("reasoner")

        self.assertIsNotNone(invalid_issue)
        artifact = {
            "question_scope": "What is the capital of France?",
            "answer_slot": {
                "entity": "France",
                "relation": "capital",
                "answer_type": "city",
                "qualifiers": [],
                "proposition_index": 0,
                "answer_field": "object_or_attribute_value",
            },
            "evidence_propositions": [
                {
                    "subject": "France",
                    "relation": "capital",
                    "object_or_attribute_value": "Paris",
                    "qualifiers": [],
                    "evidence_span": "Paris is the capital of France.",
                },
                {
                    "subject": "France",
                    "relation": "located in",
                    "object_or_attribute_value": "Europe",
                    "qualifiers": [],
                    "evidence_span": "France is a country in Europe.",
                },
            ],
            "multi_hop_chain": ["France", "capital", "Paris"],
            "candidate_answer": "Paris",
            "evidence": [
                "Paris is the capital of France.",
                "France is a country in Europe.",
            ],
        }
        env._progressive_outputs["replacement_reasoner"] = json.dumps(artifact)
        env._progressive_output_metadata["replacement_reasoner"] = {
            "tool_receipts": [
                {
                    "tool_id": "qa-retrieval",
                    "request": {"action": "read"},
                    "result": {
                        "value": {
                            "operation": "read",
                            "passage": {
                                "text": (
                                    "Paris is the capital of France. "
                                    "France is a country in Europe."
                                )
                            },
                        }
                    },
                    "error_type": None,
                }
            ]
        }

        self.assertIsNone(env._delete_admission_issue("reasoner"))

    async def test_repair_invalidation_preserves_previous_revision_artifact(self) -> None:
        graph = AgentGraph(
            [AgentNode("source", "balanced", "produce evidence")]
        )
        env = AgentWorkflowEnv(
            make_registry(),
            _ImmediateGateway(),
            graph=graph,
            problem="question",
            recovery_policy="preserve_diagnose_repair_augment",
        )
        env._progressive_outputs["source"] = "validated evidence"
        env._progressive_output_metadata["source"] = {"receipt": "revision-1"}

        env._invalidate_progressive_outputs(
            {"source"},
            current_agent_ids={"source"},
        )

        self.assertNotIn("source", env._progressive_outputs)
        self.assertEqual(
            "validated evidence",
            env._previous_revision_outputs["source"],
        )
        self.assertEqual(
            ["source"],
            env.recovery_state()["previous_revision_preserved_agent_ids"],
        )

    async def test_recovery_policy_requires_output_handoff_before_delete(self) -> None:
        registry = make_registry()
        graph = AgentGraph(
            [AgentNode("out", "balanced", "answer", role_family="format")],
            output_agent_id="out",
        )
        env = AgentWorkflowEnv(
            registry,
            _ImmediateGateway(),
            graph=graph,
            problem="question",
            recovery_policy="preserve_diagnose_repair_augment",
        )

        rejected = await env.step('{"action":"delete_agent","agent_id":"out"}')

        self.assertFalse(rejected.accepted)
        self.assertIn("Output Agent identity", rejected.feedback)
        self.assertTrue(env.graph.has_node("out"))

    async def test_recovery_policy_allows_terminal_unreachable_redundancy_after_takeover(
        self,
    ) -> None:
        registry = make_registry()
        graph = AgentGraph(
            [
                AgentNode(
                    "out",
                    "fast",
                    "active output",
                    role_family="format",
                    artifact_type="answer",
                ),
                AgentNode(
                    "redundant",
                    "cheap",
                    "redundant output",
                    role_family="format",
                    artifact_type="answer",
                ),
            ],
            output_agent_id="out",
        )
        env = AgentWorkflowEnv(
            registry,
            _ImmediateGateway(),
            graph=graph,
            problem="question",
            execute_on_edit=True,
            recovery_policy="preserve_diagnose_repair_augment",
        )
        executed = await env.step(
            '{"action":"modify_agent","agent_id":"redundant",'
            '"contract":"redundant output artifact"}'
        )
        self.assertTrue(executed.accepted)
        self.assertEqual(
            ["redundant"],
            env.recovery_state()["terminal_unreachable_agent_ids"],
        )
        self.assertIn("redundant", env.recovery_state()["preserved_agent_ids"])

        deleted = await env.step(
            '{"action":"delete_agent","agent_id":"redundant"}'
        )

        self.assertTrue(deleted.accepted)
        self.assertFalse(env.graph.has_node("redundant"))
        self.assertTrue(env.graph.has_node("out"))

    async def test_recovery_policy_keeps_successful_reachable_lineage(self) -> None:
        registry = make_registry()
        graph = AgentGraph(
            [
                AgentNode(
                    "source_a",
                    "balanced",
                    "evidence a",
                    role_family="evidence",
                    artifact_type="facts",
                ),
                AgentNode(
                    "source_b",
                    "cheap",
                    "evidence b",
                    role_family="evidence",
                    artifact_type="facts",
                ),
                AgentNode("out", "fast", "answer", role_family="output"),
            ],
            [
                AgentRelation("source_a", "out", True, False),
                AgentRelation("source_b", "out", True, False),
            ],
            output_agent_id="out",
        )
        env = AgentWorkflowEnv(
            registry,
            _ImmediateGateway(),
            graph=graph,
            problem="question",
            execute_on_edit=True,
            recovery_policy="preserve_diagnose_repair_augment",
        )
        executed = await env.step(
            '{"action":"modify_agent","agent_id":"source_a",'
            '"contract":"evidence a repaired"}'
        )
        self.assertTrue(executed.accepted)
        self.assertEqual([], env.recovery_state()["terminal_unreachable_agent_ids"])

        rejected = await env.step(
            '{"action":"delete_agent","agent_id":"source_a"}'
        )

        self.assertFalse(rejected.accepted)
        self.assertIn("not been diagnosed unusable", rejected.feedback)
        self.assertTrue(env.graph.has_node("source_a"))

    async def test_semantic_and_recovery_configuration_forks_and_defaults_are_legacy(self) -> None:
        registry = make_registry()
        default_graph = AgentGraph(
            [
                AgentNode("source", "balanced", "source"),
                AgentNode("out", "fast", "out"),
            ],
            [AgentRelation("source", "out", True, False)],
            output_agent_id="out",
        )
        legacy = AgentWorkflowEnv(
            registry,
            _ImmediateGateway(),
            graph=default_graph,
            problem="question",
        )
        deleted = await legacy.step('{"action":"delete_agent","agent_id":"source"}')
        self.assertTrue(deleted.accepted)

        configured = AgentWorkflowEnv(
            registry,
            _ImmediateGateway(),
            problem="question",
            semantic_protocol="hotpotqa_verified_answer_slot_v1",
            recovery_policy="preserve_diagnose_repair_augment",
            required_evidence_tool_id="qa-retrieval",
        )
        fork = configured.fork()
        self.assertEqual(configured.semantic_protocol, fork.semantic_protocol)
        self.assertEqual(configured.recovery_policy, fork.recovery_policy)
        self.assertEqual(
            configured.required_evidence_tool_id,
            fork.required_evidence_tool_id,
        )


if __name__ == "__main__":
    unittest.main()
