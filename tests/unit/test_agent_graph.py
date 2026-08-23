from __future__ import annotations

import asyncio
import json
import os
import random
import tempfile
import unittest
from unittest.mock import patch

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
    AgentFailureRecord,
    AgentRequest,
    AgentResponse,
    AgentRuntime,
    AgentRuntimeError,
    AgentRuntimeResult,
    ExecutionPhase,
    ReasoningExecutionAdapter,
)
from src.interactive.agent_workflow_env import (
    AgentWorkflowEnv,
    AgentWorkflowStateError,
    _HOTPOTQA_FORMAT_CONTRACT,
    _evidence_span_matches_read,
)
from src.interactive.model_registry import (
    ModelRegistry,
    ModelRegistryError,
    ModelSpec,
    ProviderSpec,
)
from src.interactive.qa_tool_adapter import (
    QA_RETRIEVAL_TOOL_ID,
    build_qa_tool_registry,
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


def make_multi_provider_registry() -> ModelRegistry:
    """Catalog fixture with both same-provider and cross-provider repair arms."""

    return ModelRegistry(
        [
            ProviderSpec("provider-a", kind="test"),
            ProviderSpec("provider-b", kind="test"),
        ],
        [
            ModelSpec("balanced", "provider-a"),
            ModelSpec("cheap", "provider-a"),
            ModelSpec("fast", "provider-b"),
            ModelSpec("alternate", "provider-b"),
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
            exc = RuntimeError("unusable executor node")
            exc.node_unusable = True
            raise exc
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
                allowed_tools=(QA_RETRIEVAL_TOOL_ID,),
                execution_mode="react",
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
                _HOTPOTQA_FORMAT_CONTRACT,
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


class _HotpotNoopRetrievalIndex:
    manifest = type(
        "Manifest",
        (),
        {
            "corpus_name": "test-public-wikipedia",
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
        raise AssertionError(f"unexpected test retrieval read {passage_id!r}")


def _hotpot_semantic_runtime(
    registry: ModelRegistry,
    gateway: _ImmediateGateway,
) -> AgentRuntime:
    """Test fixture for a semantic graph whose real Reasoner mode is ReAct."""

    return AgentRuntime(
        registry,
        gateway,
        execution_adapters={"react": ReasoningExecutionAdapter(gateway)},
        tool_registry=build_qa_tool_registry(_HotpotNoopRetrievalIndex()),
        dataset_id="hotpotqa",
        semantic_protocol="hotpotqa_verified_answer_slot_v1",
    )


def _test_read_receipt(passage_id: str) -> dict[str, object]:
    return {
        "tool_id": QA_RETRIEVAL_TOOL_ID,
        "tool_version": "test-v1",
        "request": {
            "action": "read",
            "arguments": {"passage_id": passage_id},
        },
        "result": {
            "value": {
                "operation": "read",
                "passage": {
                    "id": passage_id,
                    "text": "Paris is the capital of France.",
                },
            },
            "completed": True,
        },
        "error_type": None,
    }


def _react_exhaustion_record(
    graph: AgentGraph,
    *,
    request_id: str,
    receipts: tuple[dict[str, object], ...] = (),
    tool_plan_exhausted: bool = True,
    trace_length: int = 1,
) -> AgentFailureRecord:
    return AgentFailureRecord(
        request_id=request_id,
        agent_id="reasoner",
        phase=ExecutionPhase.SINGLE,
        graph_revision=graph.revision,
        error_type="ReactExecutionError",
        message="react agent 'reasoner' exhausted 8 turns",
        metadata={
            "react_trace": [
                {
                    "turn": turn,
                    "observation_status": "schema_invalid",
                    "public_error_code": "completion_schema_invalid",
                }
                for turn in range(1, trace_length + 1)
            ],
            "tool_receipts": list(receipts),
            "tool_plan_exhausted": tool_plan_exhausted,
        },
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
                        "answer_type": "short_answer",
                        "answer_cardinality": "single",
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
                    "alias_binding_correct": True,
                    "answer_type_cardinality_correct": True,
                    "multi_hop_complete": True,
                    "minimal_answer_surface": True,
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

    async def test_execution_failure_feedback_attributes_provider_and_repair(self) -> None:
        graph = AgentGraph(
            [AgentNode("source", "balanced", "produce evidence")],
            output_agent_id="source",
        )
        env = AgentWorkflowEnv(
            make_registry(),
            _ImmediateGateway(),
            graph=graph,
            problem="question",
            recovery_policy="preserve_diagnose_repair_augment",
        )
        failure = AgentRuntimeError(
            "gateway failed",
            failure_records=(
                AgentFailureRecord(
                    request_id="request-1",
                    agent_id="source",
                    phase=ExecutionPhase.SINGLE,
                    graph_revision=graph.revision,
                    error_type="OpenAICompatibleGatewayError",
                    message="provider request failed for fake: HTTP Error 429",
                ),
            ),
        )

        feedback = json.loads(env._execution_error_feedback(failure).split("=", 1)[1])

        attributed = feedback["failed_agents"][0]
        self.assertEqual("balanced", attributed["model_id"])
        self.assertEqual("fake", attributed["provider_id"])
        self.assertEqual("provider_request_failure", attributed["failure_category"])
        self.assertEqual("modify_agent", attributed["preferred_repair"]["action"])
        self.assertNotIn("avoid_provider_id", attributed["preferred_repair"])
        self.assertEqual(
            "fake",
            attributed["preferred_repair"]["fallback_provider_id"],
        )

    async def test_hotpot_transient_provider_repair_is_cross_provider_model_only(
        self,
    ) -> None:
        registry = make_multi_provider_registry()
        gateway = _HotpotSemanticGateway()
        graph = _hotpot_semantic_graph()
        env = AgentWorkflowEnv(
            registry,
            runtime=_hotpot_semantic_runtime(registry, gateway),
            graph=graph,
            problem="What is the capital of France?",
            semantic_protocol="hotpotqa_verified_answer_slot_v1",
            recovery_policy="preserve_diagnose_repair_augment",
            required_evidence_tool_id=QA_RETRIEVAL_TOOL_ID,
        )
        env._record_failure_state(
            (
                AgentFailureRecord(
                    request_id="request-429",
                    agent_id="reasoner",
                    phase=ExecutionPhase.SINGLE,
                    graph_revision=graph.revision,
                    error_type="OpenAICompatibleGatewayError",
                    message="provider request failed with HTTP status 429",
                ),
            ),
            current_agent_ids={node.id for node in graph.nodes},
        )

        candidate = env.model_admissible_action_targets()["modify_agent"][
            "per_agent_candidates"
        ][0]
        self.assertEqual("reasoner", candidate["agent_id"])
        self.assertEqual(["model_id"], candidate["mutable_fields"])
        self.assertEqual(
            ["alternate", "fast"],
            candidate["discrete_value_domains"]["model_id"],
        )
        self.assertEqual("provider-a", candidate["avoid_provider_id"])

        revision = env.revision
        request_count = len(gateway.requests)
        for action in (
            {
                "action": "modify_agent",
                "agent_id": "reasoner",
                "contract": "change the reasoning contract",
            },
            {
                "action": "modify_agent",
                "agent_id": "reasoner",
                "model_id": "fast",
                "contract": "change two fields",
            },
            {
                "action": "modify_agent",
                "agent_id": "reasoner",
                "model_id": "cheap",
            },
        ):
            rejected = await env.step(json.dumps(action))
            self.assertFalse(rejected.accepted)
            self.assertEqual(revision, env.revision)
            self.assertEqual(request_count, len(gateway.requests))
            self.assertEqual("balanced", env.graph.get_node("reasoner").model_id)

        repaired = await env.step(
            '{"action":"modify_agent","agent_id":"reasoner",'
            '"model_id":"fast"}'
        )
        self.assertTrue(repaired.accepted)
        self.assertNotIn("reasoner", env.recovery_state()["failed_agent_ids"])
        self.assertNotEqual(("modify_agent",), env.model_admissible_action_types())

    async def test_hotpot_transient_provider_repair_falls_back_within_provider(
        self,
    ) -> None:
        registry = make_registry()

        def make_env() -> AgentWorkflowEnv:
            return AgentWorkflowEnv(
                registry,
                runtime=_hotpot_semantic_runtime(registry, _ImmediateGateway()),
                graph=_hotpot_semantic_graph(),
                problem="What is the capital of France?",
                execute_on_edit=False,
                semantic_protocol="hotpotqa_verified_answer_slot_v1",
                recovery_policy="preserve_diagnose_repair_augment",
                required_evidence_tool_id=QA_RETRIEVAL_TOOL_ID,
            )

        normal = make_env()
        normal_candidate = next(
            item
            for item in normal.model_admissible_action_targets()["modify_agent"][
                "per_agent_candidates"
            ]
            if item["agent_id"] == "reasoner"
        )
        self.assertEqual(
            ["cheap", "fast"],
            normal_candidate["discrete_value_domains"]["model_id"],
        )

        failed = make_env()
        failed._record_failure_state(
            (
                AgentFailureRecord(
                    request_id="request-429",
                    agent_id="reasoner",
                    phase=ExecutionPhase.SINGLE,
                    graph_revision=failed.graph.revision,
                    error_type="OpenAICompatibleGatewayError",
                    message="provider request failed with HTTP status 429",
                ),
            ),
            current_agent_ids={node.id for node in failed.graph.nodes},
        )
        fallback_candidate = failed.model_admissible_action_targets()[
            "modify_agent"
        ]["per_agent_candidates"][0]
        self.assertEqual(["model_id"], fallback_candidate["mutable_fields"])
        self.assertEqual(
            ["cheap", "fast"],
            fallback_candidate["discrete_value_domains"]["model_id"],
        )
        self.assertNotIn("avoid_provider_id", fallback_candidate)

    async def test_hotpot_permanent_provider_failure_repair_is_model_only(
        self,
    ) -> None:
        registry = make_multi_provider_registry()
        graph = _hotpot_semantic_graph()
        env = AgentWorkflowEnv(
            registry,
            runtime=_hotpot_semantic_runtime(
                registry,
                _HotpotSemanticGateway(),
            ),
            graph=graph,
            problem="What is the capital of France?",
            execute_on_edit=False,
            semantic_protocol="hotpotqa_verified_answer_slot_v1",
            recovery_policy="preserve_diagnose_repair_augment",
            required_evidence_tool_id=QA_RETRIEVAL_TOOL_ID,
        )
        record = AgentFailureRecord(
            request_id="request-403",
            agent_id="reasoner",
            phase=ExecutionPhase.SINGLE,
            graph_revision=graph.revision,
            error_type="OpenAICompatibleGatewayError",
            message="provider request failed with HTTP status 403",
        )
        env._record_failure_state(
            (record,),
            current_agent_ids={node.id for node in graph.nodes},
        )

        failure = AgentRuntimeError(
            "gateway failed",
            failure_records=(record,),
        )
        feedback = json.loads(
            env._execution_error_feedback(failure).split("=", 1)[1]
        )
        attributed = feedback["failed_agents"][0]
        self.assertEqual(403, attributed["http_status"])
        self.assertEqual(
            "permanent_configuration",
            attributed["retryability"],
        )
        self.assertEqual(
            ["alternate", "fast"],
            attributed["preferred_repair"]["admitted_model_ids"],
        )
        self.assertEqual(
            "provider-a",
            attributed["preferred_repair"]["avoid_provider_id"],
        )

        candidate = env.model_admissible_action_targets()["modify_agent"][
            "per_agent_candidates"
        ][0]
        self.assertEqual("reasoner", candidate["agent_id"])
        self.assertEqual(["model_id"], candidate["mutable_fields"])
        self.assertEqual(
            ["alternate", "fast"],
            candidate["discrete_value_domains"]["model_id"],
        )
        self.assertEqual("provider-a", candidate["avoid_provider_id"])

        revision = env.revision
        rejection_feedback: list[str] = []
        for action in (
            {
                "action": "modify_agent",
                "agent_id": "reasoner",
                "contract": "change the reasoning contract",
            },
            {
                "action": "modify_agent",
                "agent_id": "reasoner",
                "model_id": "fast",
                "contract": "change two fields",
            },
            {
                "action": "modify_agent",
                "agent_id": "reasoner",
                "model_id": "cheap",
            },
        ):
            rejected = await env.step(json.dumps(action))
            self.assertFalse(rejected.accepted)
            rejection_feedback.append(rejected.feedback)
            self.assertEqual(revision, env.revision)
            self.assertEqual(
                "balanced",
                env.graph.get_node("reasoner").model_id,
            )
        self.assertIn(
            "provider failure repair",
            " ".join(rejection_feedback),
        )
        self.assertIn(
            "provider repair model_id must come from the live admitted domain",
            " ".join(rejection_feedback),
        )

        repaired = await env.step(
            '{"action":"modify_agent","agent_id":"reasoner",'
            '"model_id":"fast"}'
        )
        self.assertTrue(repaired.accepted)
        self.assertEqual("fast", env.graph.get_node("reasoner").model_id)

    async def test_provider_repair_precedes_incomplete_semantic_spine(self) -> None:
        registry = make_multi_provider_registry()
        graph = AgentGraph(
            [
                AgentNode(
                    "reasoner",
                    "balanced",
                    "retrieve and bind the answer",
                    role_family="reasoner",
                    allowed_tools=(QA_RETRIEVAL_TOOL_ID,),
                    execution_mode="react",
                ),
                AgentNode(
                    "verifier",
                    "balanced",
                    "verify the candidate",
                    role_family="verifier",
                ),
                AgentNode(
                    "formatter",
                    "fast",
                    _HOTPOTQA_FORMAT_CONTRACT,
                    role_family="format",
                ),
            ],
            [AgentRelation("reasoner", "verifier", True, False)],
        )
        env = AgentWorkflowEnv(
            registry,
            runtime=_hotpot_semantic_runtime(registry, _ImmediateGateway()),
            graph=graph,
            problem="What is the capital of France?",
            execute_on_edit=False,
            semantic_protocol="hotpotqa_verified_answer_slot_v1",
            recovery_policy="preserve_diagnose_repair_augment",
            required_evidence_tool_id=QA_RETRIEVAL_TOOL_ID,
        )
        env._record_failure_state(
            (
                AgentFailureRecord(
                    request_id="request-429-before-spine",
                    agent_id="reasoner",
                    phase=ExecutionPhase.SINGLE,
                    graph_revision=graph.revision,
                    error_type="OpenAICompatibleGatewayError",
                    message="provider request failed with HTTP status 429",
                ),
            ),
            current_agent_ids={node.id for node in graph.nodes},
        )

        self.assertEqual(("modify_agent",), env.model_admissible_action_types())
        repaired = await env.step(
            '{"action":"modify_agent","agent_id":"reasoner",'
            '"model_id":"fast"}'
        )

        self.assertTrue(repaired.accepted)
        self.assertEqual("fast", env.graph.get_node("reasoner").model_id)
        self.assertEqual(("set_relation",), env.model_admissible_action_types())

    async def test_provider_repair_precedes_neutral_topology_edits(self) -> None:
        registry = make_multi_provider_registry()
        graph = AgentGraph(
            [
                AgentNode("source", "balanced", "produce an artifact"),
                AgentNode("output", "fast", "return the answer"),
            ],
            [AgentRelation("source", "output", True, False)],
            output_agent_id="output",
        )
        env = AgentWorkflowEnv(
            registry,
            _ImmediateGateway(),
            graph=graph,
            problem="question",
            execute_on_edit=False,
            semantic_protocol="none",
            recovery_policy="default",
        )
        env._record_failure_state(
            (
                AgentFailureRecord(
                    request_id="request-neutral-403",
                    agent_id="source",
                    phase=ExecutionPhase.SINGLE,
                    graph_revision=graph.revision,
                    error_type="OpenAICompatibleGatewayError",
                    message="provider request failed with HTTP status 403",
                ),
            ),
            current_agent_ids={node.id for node in graph.nodes},
        )

        self.assertEqual(("modify_agent",), env.model_admissible_action_types())
        targets = env.model_admissible_action_targets()["modify_agent"]
        self.assertEqual(["source"], targets["agent_ids"])
        self.assertEqual(
            ["model_id"],
            targets["per_agent_candidates"][0]["mutable_fields"],
        )
        self.assertEqual(
            ["alternate", "fast"],
            targets["per_agent_candidates"][0]["discrete_value_domains"][
                "model_id"
            ],
        )

        unrelated = await env.step(
            '{"action":"set_relation","source_id":"source",'
            '"target_id":"output","source_to_target":false,'
            '"target_to_source":true}'
        )
        self.assertFalse(unrelated.accepted)
        self.assertIn("provider_repair_agent_ids", unrelated.feedback)
        contract_change = await env.step(
            '{"action":"modify_agent","agent_id":"source",'
            '"contract":"replace the contract"}'
        )
        self.assertFalse(contract_change.accepted)
        self.assertIn("must modify only model_id", contract_change.feedback)

        repaired = await env.step(
            '{"action":"modify_agent","agent_id":"source",'
            '"model_id":"fast"}'
        )
        self.assertTrue(repaired.accepted)
        self.assertEqual("fast", env.graph.get_node("source").model_id)
        self.assertNotEqual(("modify_agent",), env.model_admissible_action_types())

    async def test_neutral_output_assignment_keeps_quotient_sink(self) -> None:
        graph = AgentGraph(
            [
                AgentNode("agent-1", "cheap", "first"),
                AgentNode("agent-2", "cheap", "second"),
                AgentNode("output", "fast", "candidate output"),
            ],
            [
                AgentRelation("agent-1", "output", False, True),
                AgentRelation("agent-2", "output", False, True),
                AgentRelation("agent-1", "agent-2", False, True),
            ],
        )
        env = AgentWorkflowEnv(
            make_registry(),
            _ImmediateGateway(),
            graph=graph,
            problem="question",
            execute_on_edit=False,
            semantic_protocol="none",
            recovery_policy="default",
        )

        self.assertEqual(
            ["agent-1"],
            env.model_admissible_action_targets()["set_output"]["agent_ids"],
        )
        root_output = await env.step(
            '{"action":"set_output","agent_id":"output"}'
        )
        self.assertFalse(root_output.accepted)
        self.assertIn("quotient-graph sink", root_output.feedback)
        self.assertIsNone(env.graph.output_agent_id)

        sink_output = await env.step(
            '{"action":"set_output","agent_id":"agent-1"}'
        )
        self.assertTrue(sink_output.accepted)
        self.assertEqual("agent-1", env.graph.output_agent_id)

        reverse_output_edge = await env.step(
            '{"action":"set_relation","source_id":"agent-1",'
            '"target_id":"agent-2","source_to_target":true,'
            '"target_to_source":false}'
        )
        self.assertFalse(reverse_output_edge.accepted)
        self.assertIn("quotient-graph sink", reverse_output_edge.feedback)

    async def test_neutral_terminal_reachability_requires_exact_progress(self) -> None:
        graph = AgentGraph(
            [
                AgentNode("a", "cheap", "first source"),
                AgentNode("b", "cheap", "second source"),
                AgentNode("orphan", "cheap", "unrouted source"),
                AgentNode("out", "fast", "terminal answer"),
            ],
            [
                AgentRelation("a", "out", True, False),
                AgentRelation("b", "out", True, False),
            ],
            output_agent_id="out",
        )
        env = AgentWorkflowEnv(
            make_registry(),
            _ImmediateGateway(),
            graph=graph,
            problem="question",
            execute_on_edit=False,
            semantic_protocol="none",
            recovery_policy="default",
        )

        self.assertEqual(("set_relation",), env.model_admissible_action_types())
        candidates = env.model_admissible_action_targets()["set_relation"][
            "candidates"
        ]
        self.assertTrue(candidates)
        unrelated = await env.step(
            '{"action":"set_relation","source_id":"a",'
            '"target_id":"b","source_to_target":true,'
            '"target_to_source":false}'
        )
        self.assertFalse(unrelated.accepted)
        self.assertIn("admissible_relation_candidates", unrelated.feedback)

        exact = await env.step(json.dumps({"action": "set_relation", **candidates[0]}))
        self.assertTrue(exact.accepted)
        self.assertNotEqual(
            "graph_validation",
            env.finish_admissibility().get("stage"),
        )

    async def test_react_exhaustion_feedback_preserves_compact_public_diagnosis(
        self,
    ) -> None:
        graph = _hotpot_semantic_graph()
        registry = make_registry()
        env = AgentWorkflowEnv(
            registry,
            runtime=_hotpot_semantic_runtime(registry, _ImmediateGateway()),
            graph=graph,
            problem="What is the capital of France?",
            semantic_protocol="hotpotqa_verified_answer_slot_v1",
            recovery_policy="preserve_diagnose_repair_augment",
            required_evidence_tool_id="qa-retrieval",
        )
        passage = "Paris is the capital of France. " + ("long evidence " * 80)
        read_receipt = {
            "tool_id": "qa-retrieval",
            "tool_version": "test-v1",
            "request": {
                "action": "read",
                "arguments": {"passage_id": "p1"},
            },
            "result": {
                "value": {
                    "operation": "read",
                    "passage": {"id": "p1", "text": passage},
                },
                "completed": True,
            },
            "error_type": None,
        }
        record = AgentFailureRecord(
            request_id="request-react",
            agent_id="reasoner",
            phase=ExecutionPhase.SINGLE,
            graph_revision=graph.revision,
            error_type="ReactExecutionError",
            message="react agent 'reasoner' exhausted 8 turns",
            metadata={
                "react_trace": [
                    {
                        "turn": 1,
                        "observation": {
                            "observation_status": "success",
                            "result": {"passage": {"text": passage}},
                        },
                    },
                    {
                        "turn": 2,
                        "observation_status": "schema_invalid",
                        "public_error_code": "completion_schema_invalid",
                    },
                    {
                        "turn": 3,
                        "observation_status": "schema_invalid",
                        "public_error_code": "completion_schema_invalid",
                    },
                    {
                        "turn": 4,
                        "observation_status": "schema_invalid",
                        "public_error_code": "completion_artifact_empty",
                        "repair_instruction": (
                            "Remove action_envelope and place arguments, kind, name, "
                            "resource_id, and skill_id at the top level."
                        ),
                    },
                ],
                "tool_receipts": [read_receipt],
            },
        )
        failure = AgentRuntimeError(
            "bounded ReAct execution failed",
            failure_records=(record,),
        )

        feedback_text = env._execution_error_feedback(failure)
        feedback = json.loads(feedback_text.split("=", 1)[1])

        attributed = feedback["failed_agents"][0]
        self.assertEqual("react_turn_exhaustion", attributed["failure_category"])
        self.assertEqual("contract", attributed["preferred_repair"]["field"])
        self.assertEqual(
            "completion_condition",
            attributed["preferred_repair"]["optional_field"],
        )
        self.assertNotIn("avoid_provider_id", attributed["preferred_repair"])
        summary = attributed["react_public_error_summary"]
        self.assertEqual(4, summary["react_turn_count"])
        self.assertEqual(
            3,
            summary["observation_status_counts"]["schema_invalid"],
        )
        self.assertEqual(
            2,
            summary["public_error_code_counts"]["completion_schema_invalid"],
        )
        self.assertEqual(
            {
                "observation_status": "schema_invalid",
                "public_error_code": "completion_artifact_empty",
                "repair_instruction": (
                    "Remove action_envelope and place arguments, kind, name, "
                    "resource_id, and skill_id at the top level."
                ),
            },
            summary["last_public_error"],
        )
        self.assertEqual(1, summary["successful_tool_receipt_count"])
        self.assertEqual(1, summary["successful_evidence_read_count"])
        self.assertNotIn(passage, feedback_text)

    async def test_hotpot_recovery_requires_modify_before_augment(
        self,
    ) -> None:
        graph = _hotpot_semantic_graph()
        registry = make_registry()
        env = AgentWorkflowEnv(
            registry,
            runtime=_hotpot_semantic_runtime(registry, _ImmediateGateway()),
            graph=graph,
            problem="What is the capital of France?",
            semantic_protocol="hotpotqa_verified_answer_slot_v1",
            recovery_policy="preserve_diagnose_repair_augment",
            required_evidence_tool_id="qa-retrieval",
        )
        record = AgentFailureRecord(
            request_id="request-react",
            agent_id="reasoner",
            phase=ExecutionPhase.SINGLE,
            graph_revision=graph.revision,
            error_type="ReactExecutionError",
            message="react agent 'reasoner' exhausted 8 turns",
            metadata={
                "react_trace": [
                    {
                        "turn": 8,
                        "observation_status": "schema_invalid",
                        "public_error_code": "completion_schema_invalid",
                    }
                ]
            },
        )
        provider_record = AgentFailureRecord(
            request_id="request-provider",
            agent_id="reader",
            phase=ExecutionPhase.SINGLE,
            graph_revision=graph.revision,
            error_type="OpenAICompatibleGatewayError",
            message="provider request failed with HTTP status 429",
        )
        env._record_failure_state(
            (record, provider_record),
            current_agent_ids={node.id for node in graph.nodes},
        )

        targets = env.model_admissible_action_targets()

        modify_domain = targets["modify_agent"]
        self.assertEqual(
            ["reader", "reasoner"],
            modify_domain["agent_ids"],
        )
        self.assertIn("model_id", modify_domain["mutable_fields"])
        per_agent = {
            item["agent_id"]: item
            for item in modify_domain["per_agent_candidates"]
        }
        self.assertEqual(
            ["contract", "completion_condition"],
            per_agent["reasoner"]["mutable_fields"],
        )
        self.assertEqual({}, per_agent["reasoner"]["discrete_value_domains"])
        self.assertIn("model_id", per_agent["reader"]["mutable_fields"])
        self.assertIn("model_id", per_agent["reader"]["discrete_value_domains"])
        self.assertEqual(
            ["reader", "reasoner"],
            targets["modify_agent"]["responsible_agent_ids"],
        )
        self.assertEqual(("modify_agent",), env.model_admissible_action_types())
        self.assertEqual(
            ["modify_agent"],
            env.recovery_state()["preferred_actions"],
        )

        revision = env.revision
        for invalid_repair in (
            {
                "action": "modify_agent",
                "agent_id": "reasoner",
                "model_id": "fast",
            },
            {
                "action": "modify_agent",
                "agent_id": "reasoner",
                "artifact_type": "changed artifact",
            },
            {
                "action": "modify_agent",
                "agent_id": "reasoner",
                "contract": "repair contract",
                "completion_condition": "repair completion",
            },
        ):
            rejected = await env.step(json.dumps(invalid_repair))
            self.assertFalse(rejected.accepted)
            self.assertEqual(revision, env.revision)
        self.assertNotIn("add_subgraph", targets)
        self.assertEqual(
            ["reasoner"],
            env.recovery_state()["react_turn_exhausted_agent_ids"],
        )
        self.assertEqual(
            "single",
            env._failure_continuations["reasoner"]["execution_phase"],
        )
        self.assertEqual(
            "completion_schema_invalid",
            env._failure_continuations["reasoner"]["react_trace"][0][
                "public_error_code"
            ],
        )

        revision = env.revision
        node_ids = tuple(node.id for node in env.graph.nodes)
        request_count = len(env.runtime.gateway.requests)
        augmentation = await env.step(
            json.dumps(
                {
                    "action": "add_subgraph",
                    "agents": [
                        {
                            "agent_id": "repair_node",
                            "model_id": "balanced",
                            "contract": "diagnose the existing semantic contract",
                            "role_family": "repair",
                            "allowed_tools": [],
                            "execution_mode": "reasoning",
                        }
                    ],
                    "relations": [],
                }
            )
        )
        self.assertFalse(augmentation.accepted)
        self.assertIn("mandatory_repair_agent_ids", augmentation.feedback)
        self.assertEqual(revision, env.revision)
        self.assertEqual(node_ids, tuple(node.id for node in env.graph.nodes))
        self.assertEqual(request_count, len(env.runtime.gateway.requests))

        env._mark_agents_recovered({"reasoner"})
        self.assertNotIn("reasoner", env._failure_continuations)
        self.assertNotIn("reasoner", env.recovery_state()["failed_agent_ids"])
        self.assertIn("reader", env.recovery_state()["failed_agent_ids"])

    async def test_hotpot_react_repair_exhaustion_opens_preserving_augmentation(
        self,
    ) -> None:
        graph = _hotpot_semantic_graph()
        registry = make_registry()
        env = AgentWorkflowEnv(
            registry,
            runtime=_hotpot_semantic_runtime(registry, _ImmediateGateway()),
            graph=graph,
            problem="What is the capital of France?",
            execute_on_edit=False,
            semantic_protocol="hotpotqa_verified_answer_slot_v1",
            recovery_policy="preserve_diagnose_repair_augment",
            required_evidence_tool_id=QA_RETRIEVAL_TOOL_ID,
        )
        receipt = _test_read_receipt("p1")
        env._record_failure_state(
            (
                _react_exhaustion_record(
                    graph,
                    request_id="react-before-repair",
                    receipts=(receipt,),
                ),
            ),
            current_agent_ids={node.id for node in graph.nodes},
        )
        self.assertEqual(("modify_agent",), env.model_admissible_action_types())

        repaired_contract = (
            "align evidence to the answer slot and terminate after a valid "
            "semantic candidate"
        )
        repaired = await env.step(
            json.dumps(
                {
                    "action": "modify_agent",
                    "agent_id": "reasoner",
                    "contract": repaired_contract,
                }
            )
        )
        self.assertTrue(repaired.accepted)
        self.assertEqual(
            repaired_contract,
            env.graph.get_node("reasoner").contract,
        )
        env._record_failure_state(
            (
                _react_exhaustion_record(
                    env.graph,
                    request_id="react-after-repair",
                    receipts=(receipt,),
                    tool_plan_exhausted=False,
                    # Repeated model turns without a new Tool receipt are not
                    # retrieval progress and must open augmentation even when
                    # unused Tool-call budget remains.
                    trace_length=3,
                ),
            ),
            current_agent_ids={node.id for node in env.graph.nodes},
        )

        recovery = env.recovery_state()
        self.assertEqual(["reasoner"], recovery["repair_exhausted_agent_ids"])
        self.assertEqual([], recovery["mandatory_repair_agent_ids"])
        self.assertEqual("augment", recovery["phase"])
        self.assertIn("add_subgraph", recovery["preferred_actions"])
        self.assertNotEqual(("modify_agent",), env.model_admissible_action_types())
        add_domain = env.model_admissible_action_targets()["add_subgraph"]
        self.assertEqual(1, add_domain["min_new_agents"])
        self.assertEqual(1, add_domain["max_new_agents"])
        self.assertEqual(
            ["evidence_retriever", "repair"],
            add_domain["admitted_new_role_families"],
        )
        self.assertIn(
            "evidence_retriever",
            add_domain["admitted_new_role_families"],
        )
        self.assertIn("repair", add_domain["admitted_new_role_families"])

        deletion = await env.step(
            '{"action":"delete_agent","agent_id":"reasoner"}'
        )
        self.assertFalse(deletion.accepted)
        self.assertTrue(env.graph.has_node("reasoner"))
        self.assertEqual(
            (True, False),
            (
                env.graph.relation_bits("reasoner", "verifier").source_to_target,
                env.graph.relation_bits("reasoner", "verifier").target_to_source,
            ),
        )
        continuation = env._failure_continuations["reasoner"]
        self.assertEqual("single", continuation["execution_phase"])
        self.assertEqual(1, len(continuation["tool_receipts"]))

        # A later same-diagnosis failure during augmentation must not reopen
        # the already exhausted mandatory MODIFY loop.
        env._record_failure_state(
            (
                _react_exhaustion_record(
                    env.graph,
                    request_id="react-during-augmentation",
                    receipts=(receipt,),
                    trace_length=4,
                ),
            ),
            current_agent_ids={node.id for node in env.graph.nodes},
        )
        recovery = env.recovery_state()
        self.assertEqual(["reasoner"], recovery["repair_exhausted_agent_ids"])
        self.assertEqual([], recovery["mandatory_repair_agent_ids"])

    async def test_hotpot_react_repair_is_not_exhausted_when_receipts_grow(
        self,
    ) -> None:
        graph = _hotpot_semantic_graph()
        registry = make_registry()
        env = AgentWorkflowEnv(
            registry,
            runtime=_hotpot_semantic_runtime(registry, _ImmediateGateway()),
            graph=graph,
            problem="What is the capital of France?",
            execute_on_edit=False,
            semantic_protocol="hotpotqa_verified_answer_slot_v1",
            recovery_policy="preserve_diagnose_repair_augment",
            required_evidence_tool_id=QA_RETRIEVAL_TOOL_ID,
        )
        first_receipt = _test_read_receipt("p1")
        env._record_failure_state(
            (
                _react_exhaustion_record(
                    graph,
                    request_id="react-before-repair",
                    receipts=(first_receipt,),
                ),
            ),
            current_agent_ids={node.id for node in graph.nodes},
        )
        repaired = await env.step(
            '{"action":"modify_agent","agent_id":"reasoner",'
            '"completion_condition":"emit a valid semantic candidate"}'
        )
        self.assertTrue(repaired.accepted)
        env._record_failure_state(
            (
                _react_exhaustion_record(
                    env.graph,
                    request_id="react-after-repair-with-progress",
                    receipts=(first_receipt, _test_read_receipt("p2")),
                ),
            ),
            current_agent_ids={node.id for node in env.graph.nodes},
        )

        recovery = env.recovery_state()
        self.assertEqual([], recovery["repair_exhausted_agent_ids"])
        self.assertEqual(["reasoner"], recovery["mandatory_repair_agent_ids"])
        self.assertEqual(("modify_agent",), env.model_admissible_action_types())

    async def test_hotpot_recovery_continuation_handoff_is_target_scoped_and_ephemeral(
        self,
    ) -> None:
        graph = _hotpot_semantic_graph()
        registry = make_registry()
        env = AgentWorkflowEnv(
            registry,
            runtime=_hotpot_semantic_runtime(registry, _ImmediateGateway()),
            graph=graph,
            problem="What is the capital of France?",
            execute_on_edit=False,
            semantic_protocol="hotpotqa_verified_answer_slot_v1",
            recovery_policy="preserve_diagnose_repair_augment",
            required_evidence_tool_id=QA_RETRIEVAL_TOOL_ID,
        )
        successful_tool_trace = {
            "turn": 1,
            "structured_action": {
                "kind": "tool",
                "name": "read",
                "resource_id": QA_RETRIEVAL_TOOL_ID,
                "skill_id": None,
                "arguments": {"passage_id": "p1"},
            },
            "observation": {
                "observation_status": "success",
                "tool_id": QA_RETRIEVAL_TOOL_ID,
                "executed_action": {
                    "kind": "tool",
                    "name": "read",
                    "resource_id": QA_RETRIEVAL_TOOL_ID,
                    "skill_id": None,
                    "arguments": {"passage_id": "p1"},
                },
                "result": {"operation": "read", "passage_id": "p1"},
            },
        }
        role_specific_completion_error = {
            "turn": 2,
            "observation_status": "schema_invalid",
            "public_error_code": (
                "qa_semantic_evidence_provenance_invalid: Reasoner "
                "candidate_answer must occur verbatim in the selected evidence_span"
            ),
            "executed_action": {
                "kind": "complete",
                "name": "complete",
                "resource_id": None,
                "skill_id": None,
                "arguments": {"value": "invalid Reasoner artifact"},
            },
        }
        trace = [
            successful_tool_trace,
            role_specific_completion_error,
            {
                "turn": 3,
                "observation_status": "schema_invalid",
                "public_error_code": "qa_retrieval_duplicate_normalized_query",
            }
        ]
        receipts = [_test_read_receipt("p1")]
        env._repair_exhausted_agent_ids.add("reasoner")
        env._failure_continuations["reasoner"] = {
            "execution_phase": "single",
            "react_trace": trace,
            "tool_receipts": receipts,
        }
        action_payload = json.dumps(
            {
                "action": "add_subgraph",
                "agents": [
                    {
                        "agent_id": "repair_reader",
                        "model_id": "cheap",
                        "contract": "retrieve grounded evidence for the Reasoner",
                        "role_family": "evidence_retriever",
                        "allowed_tools": [QA_RETRIEVAL_TOOL_ID],
                        "execution_mode": "react",
                    }
                ],
                "relations": [],
            }
        )
        action = env.parser.parse(action_payload)

        handoff = env._recovery_continuation_handoff(action)

        self.assertEqual(["repair_reader"], list(handoff))
        projected = handoff["repair_reader"]
        self.assertEqual("single", projected["execution_phase"])
        self.assertEqual(
            "reasoner",
            projected["continuation_source_agent_id"],
        )
        self.assertEqual([successful_tool_trace], projected["react_trace"])
        self.assertEqual(receipts, projected["tool_receipts"])
        self.assertNotIn("repair_reader", env._failure_continuations)
        self.assertNotIn(
            "continuation_source_agent_id",
            env._failure_continuations["reasoner"],
        )

        observed_failure_metadata: dict[str, object] = {}

        async def capture_overlay(
            candidate_graph: AgentGraph,
            *args: object,
            **kwargs: object,
        ) -> AgentRuntimeResult:
            del args
            observed_failure_metadata.update(
                kwargs["prior_failure_metadata"]
            )
            return AgentRuntimeResult(
                run_id="handoff-overlay",
                graph_revision=candidate_graph.revision,
                output_agent_id=candidate_graph.output_agent_id,
                final_answer=None,
                outputs={},
                output_metadata={},
                calls=(),
                block_completion_order=(),
                executed_agent_ids=(),
                deferred_agent_ids=tuple(
                    node.id for node in candidate_graph.nodes
                ),
            )

        env.execute_on_edit = True
        with patch.object(env.runtime, "execute", side_effect=capture_overlay):
            result = await env.step(action_payload)

        self.assertTrue(result.accepted)
        self.assertEqual(projected, observed_failure_metadata["repair_reader"])
        self.assertEqual(
            "single",
            observed_failure_metadata["repair_reader"]["execution_phase"],
        )
        self.assertEqual(
            "reasoner",
            observed_failure_metadata["repair_reader"][
                "continuation_source_agent_id"
            ],
        )
        self.assertNotIn("repair_reader", env._failure_continuations)
        self.assertIn("reasoner", env._failure_continuations)

    def test_hotpot_recovery_continuation_handoff_fails_closed(self) -> None:
        registry = make_registry()
        graph = _hotpot_semantic_graph()
        graph.add_agent(
            AgentNode(
                "reasoner_2",
                "balanced",
                "reason over grounded evidence",
                role_family="reasoner",
                allowed_tools=(QA_RETRIEVAL_TOOL_ID,),
                execution_mode="react",
            )
        )
        env = AgentWorkflowEnv(
            registry,
            runtime=_hotpot_semantic_runtime(registry, _ImmediateGateway()),
            graph=graph,
            problem="What is the capital of France?",
            execute_on_edit=False,
            semantic_protocol="hotpotqa_verified_answer_slot_v1",
            recovery_policy="preserve_diagnose_repair_augment",
            required_evidence_tool_id=QA_RETRIEVAL_TOOL_ID,
        )
        continuation = {
            "execution_phase": "single",
            "react_trace": [{"turn": 1, "observation_status": "tool_result"}],
            "tool_receipts": [_test_read_receipt("p1")],
        }
        env._repair_exhausted_agent_ids.update({"reasoner", "reasoner_2"})
        env._failure_continuations.update(
            {
                "reasoner": continuation,
                "reasoner_2": continuation,
            }
        )

        def parse_target(role_family: str):
            return env.parser.parse(
                json.dumps(
                    {
                        "action": "add_subgraph",
                        "agents": [
                            {
                                "agent_id": "repair_reader",
                                "model_id": "cheap",
                                "contract": "repair the failed semantic path",
                                "role_family": role_family,
                                "allowed_tools": [QA_RETRIEVAL_TOOL_ID],
                                "execution_mode": "react",
                            }
                        ],
                        "relations": [],
                    }
                )
            )

        self.assertEqual(
            {},
            env._recovery_continuation_handoff(
                parse_target("evidence_retriever")
            ),
        )

        env._repair_exhausted_agent_ids.remove("reasoner_2")
        self.assertEqual(
            {},
            env._recovery_continuation_handoff(parse_target("repair")),
        )

        non_recovery = AgentWorkflowEnv(
            registry,
            runtime=AgentRuntime(registry, _ImmediateGateway()),
            graph=_hotpot_semantic_graph(),
            problem="What is the capital of France?",
            execute_on_edit=False,
            required_evidence_tool_id=QA_RETRIEVAL_TOOL_ID,
        )
        non_recovery._repair_exhausted_agent_ids.add("reasoner")
        non_recovery._failure_continuations["reasoner"] = continuation
        self.assertEqual(
            {},
            non_recovery._recovery_continuation_handoff(
                parse_target("evidence_retriever")
            ),
        )

    def test_recovery_handoff_survives_target_provider_failure(self) -> None:
        graph = _hotpot_semantic_graph()
        registry = make_registry()
        env = AgentWorkflowEnv(
            registry,
            runtime=_hotpot_semantic_runtime(registry, _ImmediateGateway()),
            graph=graph,
            problem="What is the capital of France?",
            execute_on_edit=False,
            semantic_protocol="hotpotqa_verified_answer_slot_v1",
            recovery_policy="preserve_diagnose_repair_augment",
            required_evidence_tool_id=QA_RETRIEVAL_TOOL_ID,
        )
        tool_trace = {
            "turn": 1,
            "observation": {
                "observation_status": "success",
                "tool_id": QA_RETRIEVAL_TOOL_ID,
                "executed_action": {
                    "kind": "tool",
                    "name": "read",
                    "resource_id": QA_RETRIEVAL_TOOL_ID,
                    "skill_id": None,
                    "arguments": {"passage_id": "p1"},
                },
                "result": {"operation": "read", "passage_id": "p1"},
            },
        }
        source_completion_error = {
            "turn": 2,
            "observation_status": "schema_invalid",
            "public_error_code": "completion_schema_invalid",
        }
        receipt = _test_read_receipt("p1")
        env._failure_continuations["reasoner"] = {
            "execution_phase": "single",
            "react_trace": [tool_trace, source_completion_error],
            "tool_receipts": [receipt],
        }

        env._record_failure_state(
            (
                AgentFailureRecord(
                    request_id="reader-provider-failure",
                    agent_id="reader",
                    phase=ExecutionPhase.SINGLE,
                    graph_revision=graph.revision,
                    error_type="OpenAICompatibleGatewayError",
                    message="provider request failed for reader with HTTP 403",
                    metadata={"continuation_source_agent_id": "reasoner"},
                ),
            ),
            current_agent_ids={node.id for node in graph.nodes},
        )

        retained = env._failure_continuations["reader"]
        self.assertEqual("reasoner", retained["continuation_source_agent_id"])
        self.assertEqual("single", retained["execution_phase"])
        self.assertEqual([tool_trace], retained["react_trace"])
        self.assertEqual([receipt], retained["tool_receipts"])

    async def test_hotpot_non_react_failures_do_not_exhaust_react_repair(
        self,
    ) -> None:
        for error_type, message in (
            (
                "OpenAICompatibleGatewayError",
                "provider request failed with HTTP status 429",
            ),
            (
                "AgentRuntimeError",
                "execution contract invalid after a structural edit",
            ),
        ):
            with self.subTest(error_type=error_type):
                graph = _hotpot_semantic_graph()
                registry = make_registry()
                env = AgentWorkflowEnv(
                    registry,
                    runtime=_hotpot_semantic_runtime(
                        registry, _ImmediateGateway()
                    ),
                    graph=graph,
                    problem="What is the capital of France?",
                    execute_on_edit=False,
                    semantic_protocol="hotpotqa_verified_answer_slot_v1",
                    recovery_policy="preserve_diagnose_repair_augment",
                    required_evidence_tool_id=QA_RETRIEVAL_TOOL_ID,
                )
                receipt = _test_read_receipt("p1")
                env._record_failure_state(
                    (
                        _react_exhaustion_record(
                            graph,
                            request_id="react-before-repair",
                            receipts=(receipt,),
                        ),
                    ),
                    current_agent_ids={node.id for node in graph.nodes},
                )
                repaired = await env.step(
                    '{"action":"modify_agent","agent_id":"reasoner",'
                    '"contract":"repair the completion contract"}'
                )
                self.assertTrue(repaired.accepted)
                env._record_failure_state(
                    (
                        AgentFailureRecord(
                            request_id="non-react-after-repair",
                            agent_id="reasoner",
                            phase=ExecutionPhase.SINGLE,
                            graph_revision=env.graph.revision,
                            error_type=error_type,
                            message=message,
                            metadata={
                                "tool_receipts": [receipt],
                                "tool_plan_exhausted": True,
                            },
                        ),
                    ),
                    current_agent_ids={node.id for node in env.graph.nodes},
                )

                recovery = env.recovery_state()
                self.assertEqual([], recovery["repair_exhausted_agent_ids"])
                self.assertEqual(
                    ["reasoner"], recovery["mandatory_repair_agent_ids"]
                )

    async def test_hotpot_measured_failure_is_not_attributed_to_blocked_downstream(
        self,
    ) -> None:
        graph = _hotpot_semantic_graph()
        registry = make_registry()
        env = AgentWorkflowEnv(
            registry,
            runtime=_hotpot_semantic_runtime(registry, _ImmediateGateway()),
            graph=graph,
            problem="Who founded the Meridian Archive?",
            semantic_protocol="hotpotqa_verified_answer_slot_v1",
            recovery_policy="preserve_diagnose_repair_augment",
            required_evidence_tool_id="qa-retrieval",
        )
        env._record_failure_state(
            (
                AgentFailureRecord(
                    request_id="request-reasoner",
                    agent_id="reasoner",
                    phase=ExecutionPhase.SINGLE,
                    graph_revision=graph.revision,
                    error_type="ReactExecutionError",
                    message="react agent 'reasoner' exhausted 8 turns",
                ),
            ),
            current_agent_ids={node.id for node in graph.nodes},
        )
        # Runtime pending/dirty closure contains the downstream consumers, but
        # neither one raised the measured failure.
        env._unresolved_dirty_agents.update(
            {"reasoner", "verifier", "formatter"}
        )

        modify_domain = env.model_admissible_action_targets()["modify_agent"]

        self.assertEqual(["reasoner"], modify_domain["agent_ids"])
        self.assertEqual(["reasoner"], modify_domain["failed_agent_ids"])
        self.assertEqual(["reasoner"], modify_domain["responsible_agent_ids"])
        self.assertEqual(
            ["contract", "completion_condition"],
            modify_domain["per_agent_candidates"][0]["mutable_fields"],
        )

    async def test_hotpot_augmentation_excludes_healthy_semantic_role_duplicates(
        self,
    ) -> None:
        graph = _hotpot_semantic_graph()
        registry = make_registry()
        gateway = _ImmediateGateway()
        env = AgentWorkflowEnv(
            registry,
            runtime=_hotpot_semantic_runtime(registry, gateway),
            graph=graph,
            problem="What is the capital of France?",
            execute_on_edit=False,
            semantic_protocol="hotpotqa_verified_answer_slot_v1",
            recovery_policy="preserve_diagnose_repair_augment",
            required_evidence_tool_id="qa-retrieval",
        )

        add_domain = env.model_admissible_action_targets()["add_subgraph"]
        self.assertEqual(
            ["evidence_retriever", "repair"],
            add_domain["admitted_new_role_families"],
        )
        revision = env.revision
        for agent_id, role_family, contract in (
            ("second_verifier", "verifier", "verify the semantic candidate"),
            ("second_formatter", "format", _HOTPOTQA_FORMAT_CONTRACT),
        ):
            rejected = await env.step(
                json.dumps(
                    {
                        "action": "add_subgraph",
                        "agents": [
                            {
                                "agent_id": agent_id,
                                "model_id": "balanced",
                                "contract": contract,
                                "role_family": role_family,
                                "allowed_tools": [],
                                "execution_mode": "reasoning",
                            }
                        ],
                        "relations": [],
                    }
                )
            )
            self.assertFalse(rejected.accepted)
            self.assertIn("already owns that semantic responsibility", rejected.feedback)
            self.assertEqual(revision, env.revision)
            self.assertFalse(env.graph.has_node(agent_id))
        self.assertEqual([], gateway.requests)

    async def test_hotpot_verifier_cannot_take_formatter_role_contract(self) -> None:
        graph = _hotpot_semantic_graph()
        registry = make_registry()
        gateway = _ImmediateGateway()
        env = AgentWorkflowEnv(
            registry,
            runtime=_hotpot_semantic_runtime(registry, gateway),
            graph=graph,
            problem="What is the capital of France?",
            execute_on_edit=False,
            semantic_protocol="hotpotqa_verified_answer_slot_v1",
            recovery_policy="preserve_diagnose_repair_augment",
            required_evidence_tool_id="qa-retrieval",
        )
        revision = env.revision
        original_contract = env.graph.get_node("verifier").contract

        rejected = await env.step(
            json.dumps(
                {
                    "action": "modify_agent",
                    "agent_id": "verifier",
                    "contract": _HOTPOTQA_FORMAT_CONTRACT.capitalize() + ".",
                }
            )
        )

        self.assertFalse(rejected.accepted)
        self.assertIn("formatting-only role contract", rejected.feedback)
        self.assertEqual(revision, env.revision)
        self.assertEqual(
            original_contract,
            env.graph.get_node("verifier").contract,
        )
        self.assertEqual([], gateway.requests)

    async def test_hotpot_relation_domain_preserves_unique_formatter_predecessor(
        self,
    ) -> None:
        graph = _hotpot_semantic_graph()
        graph.add_agent(
            AgentNode(
                "reviewer",
                "balanced",
                "verify evidence binding hops and scope without changing candidate",
                role_family="verifier",
            )
        )
        registry = make_registry()
        gateway = _ImmediateGateway()
        env = AgentWorkflowEnv(
            registry,
            runtime=_hotpot_semantic_runtime(registry, gateway),
            graph=graph,
            problem="What is the capital of France?",
            execute_on_edit=False,
            semantic_protocol="hotpotqa_verified_answer_slot_v1",
            recovery_policy="preserve_diagnose_repair_augment",
            required_evidence_tool_id="qa-retrieval",
        )

        targets = env.model_admissible_action_targets()
        self.assertFalse(
            any(
                candidate["source_id"] == "reviewer"
                and candidate["target_id"] == "formatter"
                and candidate["source_to_target"] is True
                for candidate in targets["set_relation"]["candidates"]
            )
        )
        revision = env.revision
        rejected = await env.step(
            '{"action":"set_relation","source_id":"reviewer",'
            '"target_id":"formatter","source_to_target":true,'
            '"target_to_source":false}'
        )
        self.assertFalse(rejected.accepted)
        self.assertIn("exactly one upstream semantic-answer artifact", rejected.feedback)
        self.assertEqual(revision, env.revision)
        self.assertEqual([], gateway.requests)

    async def test_hotpot_repair_then_allows_fan_in_and_reciprocal_topology(
        self,
    ) -> None:
        graph = _hotpot_semantic_graph()
        registry = make_registry()
        env = AgentWorkflowEnv(
            registry,
            runtime=_hotpot_semantic_runtime(registry, _ImmediateGateway()),
            graph=graph,
            problem="What is the capital of France?",
            execute_on_edit=False,
            semantic_protocol="hotpotqa_verified_answer_slot_v1",
            recovery_policy="preserve_diagnose_repair_augment",
            required_evidence_tool_id="qa-retrieval",
        )
        env._record_failure_state(
            (
                AgentFailureRecord(
                    request_id="request-reasoner",
                    agent_id="reasoner",
                    phase=ExecutionPhase.SINGLE,
                    graph_revision=graph.revision,
                    error_type="OpenAICompatibleGatewayError",
                    message="provider request failed with HTTP status 429",
                ),
            ),
            current_agent_ids={node.id for node in graph.nodes},
        )
        self.assertEqual(("modify_agent",), env.model_admissible_action_types())

        env._mark_agents_recovered({"reasoner"})
        added = await env.step(
            json.dumps(
                {
                    "action": "add_subgraph",
                    "agents": [
                        {
                            "agent_id": "reader_2",
                            "model_id": "cheap",
                            "contract": "retrieve additional evidence for the Reasoner",
                            "role_family": "evidence_retriever",
                            "allowed_tools": ["qa-retrieval"],
                            "execution_mode": "react",
                        }
                    ],
                    "relations": [
                        {
                            "source_id": "reader_2",
                            "target_id": "reasoner",
                            "source_to_target": True,
                            "target_to_source": False,
                        }
                    ],
                }
            )
        )
        self.assertTrue(added.accepted)
        reciprocal = await env.step(
            '{"action":"set_relation","source_id":"reasoner",'
            '"target_id":"verifier","source_to_target":true,'
            '"target_to_source":true}'
        )
        self.assertTrue(reciprocal.accepted)
        topology = env.graph.topology_statistics()
        self.assertIn("fan_in", topology["topology_motifs"])
        self.assertIn("reciprocal", topology["topology_motifs"])
        self.assertIsNone(env._semantic_edit_issue_for(env.graph))

    async def test_finish_admissibility_keeps_structured_graph_diagnosis(self) -> None:
        graph = AgentGraph(
            [
                AgentNode("orphan", "cheap", "unused branch"),
                AgentNode("output", "fast", "answer"),
            ],
            output_agent_id="output",
        )
        env = AgentWorkflowEnv(
            make_registry(),
            _ImmediateGateway(),
            graph=graph,
            problem="question",
            recovery_policy="preserve_diagnose_repair_augment",
        )

        admission = env.finish_admissibility()

        self.assertFalse(admission["admissible"])
        self.assertEqual("graph_validation", admission["stage"])
        issues = admission["issues"]
        self.assertTrue(any(item["code"] == "cannot_reach_output" for item in issues))
        self.assertTrue(
            any("orphan" in item["agent_ids"] for item in issues)
        )
        self.assertEqual(
            "preserve -> diagnose -> repair -> augment",
            admission["recovery_state"]["strategy"],
        )

    async def test_hotpot_finish_attribution_prioritizes_structure_repair(self) -> None:
        registry = make_registry()
        output_missing = AgentWorkflowEnv(
            registry,
            _ImmediateGateway(),
            graph=AgentGraph(
                [
                    AgentNode(
                        "formatter",
                        "fast",
                        _HOTPOTQA_FORMAT_CONTRACT,
                        role_family="format",
                    )
                ]
            ),
            problem="What is the capital of France?",
            semantic_protocol="hotpotqa_verified_answer_slot_v1",
            recovery_policy="preserve_diagnose_repair_augment",
            required_evidence_tool_id="qa-retrieval",
        ).finish_admissibility()

        self.assertEqual("graph_validation", output_missing["stage"])
        self.assertEqual(
            "semantic_lineage_construction",
            output_missing["failure_attribution"]["responsible_constraint"],
        )
        self.assertEqual(
            "add_subgraph",
            output_missing["failure_attribution"]["preferred_action_order"][0],
        )

        disconnected = _hotpot_semantic_graph()
        disconnected.add_agent(
            AgentNode(
                "orphan",
                "cheap",
                "preserve supplementary evidence",
                role_family="evidence_retriever",
            )
        )
        unreachable = AgentWorkflowEnv(
            registry,
            _ImmediateGateway(),
            graph=disconnected,
            problem="What is the capital of France?",
            semantic_protocol="hotpotqa_verified_answer_slot_v1",
            recovery_policy="preserve_diagnose_repair_augment",
            required_evidence_tool_id="qa-retrieval",
        ).finish_admissibility()

        attribution = unreachable["failure_attribution"]
        self.assertEqual("terminal_reachability", attribution["responsible_constraint"])
        self.assertEqual(["orphan"], attribution["responsible_agent_ids"])
        self.assertEqual("set_relation", attribution["preferred_action_order"][0])
        self.assertFalse(attribution["delete_allowed_before_replacement_takeover"])

        semantic_env = AgentWorkflowEnv(
            registry,
            _ImmediateGateway(),
            graph=_hotpot_semantic_graph(),
            problem="Who won Super Bowl XX?",
            semantic_protocol="hotpotqa_verified_answer_slot_v1",
            recovery_policy="preserve_diagnose_repair_augment",
            required_evidence_tool_id=QA_RETRIEVAL_TOOL_ID,
        )
        answer_slot_attribution = semantic_env._semantic_repair_attribution(
            "Verifier field 'answer_type_cardinality_correct' must be true"
        )
        self.assertIsNotNone(answer_slot_attribution)
        assert answer_slot_attribution is not None
        self.assertEqual(
            "reasoner",
            answer_slot_attribution["responsible_role_family"],
        )
        self.assertEqual(
            "reasoner",
            answer_slot_attribution["responsible_agent_id"],
        )
        wrapped_surface_attribution = semantic_env._semantic_repair_attribution(
            "Verifier 'verifier' semantic artifact is invalid: Verifier field "
            "'minimal_answer_surface' must be true. The Reasoner candidate "
            "already passed answer-slot binding and retrieved-evidence alignment"
        )
        self.assertIsNotNone(wrapped_surface_attribution)
        assert wrapped_surface_attribution is not None
        self.assertEqual(
            "reasoner",
            wrapped_surface_attribution["responsible_role_family"],
        )
        self.assertEqual(
            "reasoner",
            wrapped_surface_attribution["responsible_agent_id"],
        )
        verifier_schema_attribution = semantic_env._semantic_repair_attribution(
            "Verifier field 'candidate_answer' must be one bare answer span"
        )
        self.assertIsNotNone(verifier_schema_attribution)
        assert verifier_schema_attribution is not None
        self.assertEqual(
            "verifier",
            verifier_schema_attribution["responsible_role_family"],
        )

    async def test_hotpot_live_domain_closes_semantic_spine_before_output(
        self,
    ) -> None:
        complete = _hotpot_semantic_graph()
        graph = AgentGraph(
            complete.nodes,
            [
                relation
                for relation in complete.relations
                if {relation.source_id, relation.target_id}
                != {"verifier", "formatter"}
            ],
        )
        registry = make_registry()
        env = AgentWorkflowEnv(
            registry,
            runtime=_hotpot_semantic_runtime(registry, _ImmediateGateway()),
            graph=graph,
            problem="What is the capital of France?",
            execute_on_edit=False,
            semantic_protocol="hotpotqa_verified_answer_slot_v1",
            recovery_policy="preserve_diagnose_repair_augment",
            required_evidence_tool_id=QA_RETRIEVAL_TOOL_ID,
        )
        env._repair_exhausted_agent_ids.add("reasoner")

        self.assertEqual(("set_relation",), env.model_admissible_action_types())
        relation_candidates = env.model_admissible_action_targets()[
            "set_relation"
        ]["candidates"]
        self.assertEqual(
            [
                {
                    "source_id": "verifier",
                    "target_id": "formatter",
                    "source_to_target": True,
                    "target_to_source": False,
                }
            ],
            relation_candidates,
        )
        premature_output = await env.step(
            '{"action":"set_output","agent_id":"formatter"}'
        )
        self.assertFalse(premature_output.accepted)
        self.assertIn("close the declared", premature_output.feedback)

        closed = await env.step(
            '{"action":"set_relation","source_id":"verifier",'
            '"target_id":"formatter","source_to_target":true,'
            '"target_to_source":false}'
        )
        self.assertTrue(closed.accepted)
        self.assertEqual(("set_output",), env.model_admissible_action_types())
        self.assertEqual(
            ["formatter"],
            env.model_admissible_action_targets()["set_output"]["agent_ids"],
        )
        selected = await env.step(
            '{"action":"set_output","agent_id":"formatter"}'
        )
        self.assertTrue(selected.accepted)
        env._repair_exhausted_agent_ids.discard("reasoner")

        removed_spine = await env.step(
            '{"action":"set_relation","source_id":"reasoner",'
            '"target_id":"verifier","source_to_target":false,'
            '"target_to_source":false}'
        )
        self.assertFalse(removed_spine.accepted)
        self.assertIn("semantic-lineage relation", removed_spine.feedback)

    async def test_hotpot_live_domain_completes_missing_semantic_role_before_augmentation(
        self,
    ) -> None:
        complete = _hotpot_semantic_graph()
        graph = AgentGraph(
            [node for node in complete.nodes if node.id != "formatter"],
            [
                relation
                for relation in complete.relations
                if "formatter" not in {relation.source_id, relation.target_id}
            ],
        )
        registry = make_registry()
        env = AgentWorkflowEnv(
            registry,
            runtime=_hotpot_semantic_runtime(registry, _ImmediateGateway()),
            graph=graph,
            problem="What is the capital of France?",
            execute_on_edit=False,
            semantic_protocol="hotpotqa_verified_answer_slot_v1",
            recovery_policy="preserve_diagnose_repair_augment",
            required_evidence_tool_id=QA_RETRIEVAL_TOOL_ID,
        )
        env._repair_exhausted_agent_ids.add("reasoner")

        self.assertEqual(("add_subgraph",), env.model_admissible_action_types())
        add_domain = env.model_admissible_action_targets()["add_subgraph"]
        self.assertEqual(1, add_domain["min_new_agents"])
        self.assertEqual(1, add_domain["max_new_agents"])
        self.assertEqual(["format"], add_domain["admitted_new_role_families"])

        wrong_role = await env.step(
            json.dumps(
                {
                    "action": "add_subgraph",
                    "agents": [
                        {
                            "agent_id": "repair_reader",
                            "model_id": "cheap",
                            "contract": "retrieve supplementary evidence",
                            "role_family": "evidence_retriever",
                            "allowed_tools": [QA_RETRIEVAL_TOOL_ID],
                            "execution_mode": "react",
                        }
                    ],
                    "relations": [],
                }
            )
        )
        self.assertFalse(wrong_role.accepted)
        self.assertIn("missing semantic responsibilities", wrong_role.feedback)

        formatter = await env.step(
            json.dumps(
                {
                    "action": "add_subgraph",
                    "agents": [
                        {
                            "agent_id": "formatter",
                            "model_id": "fast",
                            "contract": _HOTPOTQA_FORMAT_CONTRACT,
                            "role_family": "format",
                            "allowed_tools": [],
                            "execution_mode": "reasoning",
                        }
                    ],
                    "relations": [],
                }
            )
        )
        self.assertTrue(formatter.accepted)
        self.assertEqual(("set_relation",), env.model_admissible_action_types())

    async def test_hotpot_partial_auxiliary_success_preserves_reasoner_recovery(
        self,
    ) -> None:
        complete = _hotpot_semantic_graph()
        graph = AgentGraph(
            [node for node in complete.nodes if node.id != "reader"],
            [
                relation
                for relation in complete.relations
                if "reader" not in {relation.source_id, relation.target_id}
            ],
            output_agent_id="formatter",
        )
        graph.add_agent(
            AgentNode(
                "repair_evidence",
                "cheap",
                "retrieve additional evidence for the failed Reasoner",
                role_family="evidence_retriever",
                allowed_tools=(QA_RETRIEVAL_TOOL_ID,),
                execution_mode="react",
            )
        )
        registry = make_registry()
        runtime = _hotpot_semantic_runtime(registry, _ImmediateGateway())
        env = AgentWorkflowEnv(
            registry,
            runtime=runtime,
            graph=graph,
            problem="What is the capital of France?",
            execute_on_edit=True,
            semantic_protocol="hotpotqa_verified_answer_slot_v1",
            recovery_policy="preserve_diagnose_repair_augment",
            required_evidence_tool_id=QA_RETRIEVAL_TOOL_ID,
        )
        env._failed_agent_ids.add("reasoner")
        env._react_exhausted_agent_ids.add("reasoner")
        env._repair_exhausted_agent_ids.add("reasoner")
        env._failure_continuations["reasoner"] = {
            "execution_phase": "single",
            "tool_receipts": [_test_read_receipt("p1")],
        }
        env._progressive_outputs["repair_evidence"] = "retrieved evidence"

        self.assertEqual(("set_relation",), env.model_admissible_action_types())
        candidate = env.model_admissible_action_targets()["set_relation"][
            "candidates"
        ][0]
        self.assertEqual("repair_evidence", candidate["source_id"])
        self.assertEqual("reasoner", candidate["target_id"])
        self.assertTrue(candidate["source_to_target"])
        self.assertFalse(candidate["target_to_source"])

        async def partial_success(
            candidate_graph: AgentGraph,
            *args: object,
            **kwargs: object,
        ) -> AgentRuntimeResult:
            del args, kwargs
            return AgentRuntimeResult(
                run_id="partial-success",
                graph_revision=candidate_graph.revision,
                output_agent_id=candidate_graph.output_agent_id,
                final_answer=None,
                outputs={"repair_evidence": "retrieved evidence"},
                output_metadata={"repair_evidence": {}},
                calls=(),
                block_completion_order=(("repair_evidence",),),
                executed_agent_ids=("repair_evidence",),
                deferred_agent_ids=("reasoner", "verifier", "formatter"),
            )

        with patch.object(runtime, "execute", side_effect=partial_success):
            result = await env.step(json.dumps({"action": "set_relation", **candidate}))

        self.assertTrue(result.accepted)
        recovery = env.recovery_state()
        self.assertIn("reasoner", recovery["failed_agent_ids"])
        self.assertIn("reasoner", recovery["repair_exhausted_agent_ids"])
        self.assertIn("reasoner", env._failure_continuations)
        self.assertIn("reasoner", env.unresolved_dirty_agent_ids)

    async def test_repair_exhausted_auxiliary_replacement_takeover_deletes_first(
        self,
    ) -> None:
        graph = _hotpot_semantic_graph()
        graph.add_agent(
            AgentNode(
                "failed_reader",
                "cheap",
                "retrieve replacement evidence",
                role_family="evidence_retriever",
                allowed_tools=(QA_RETRIEVAL_TOOL_ID,),
                execution_mode="react",
                artifact_type="retrieval_evidence",
            )
        )
        graph.set_relation("failed_reader", "reasoner", True, False)
        registry = make_registry()
        env = AgentWorkflowEnv(
            registry,
            runtime=_hotpot_semantic_runtime(registry, _ImmediateGateway()),
            graph=graph,
            problem="What is the capital of France?",
            execute_on_edit=False,
            semantic_protocol="hotpotqa_verified_answer_slot_v1",
            recovery_policy="preserve_diagnose_repair_augment",
            required_evidence_tool_id=QA_RETRIEVAL_TOOL_ID,
        )
        env._progressive_outputs["reader"] = "grounded replacement evidence"
        env._failed_agent_ids.update({"failed_reader", "reasoner"})
        env._repair_exhausted_agent_ids.update({"failed_reader", "reasoner"})
        env._diagnosed_unusable_agent_ids.add("failed_reader")
        env._latest_failure_record_by_agent["failed_reader"] = AgentFailureRecord(
            request_id="failed-reader-repair-exhausted",
            agent_id="failed_reader",
            phase=ExecutionPhase.SINGLE,
            graph_revision=graph.revision,
            error_type="ReactExecutionError",
            message="react agent 'failed_reader' exhausted 8 turns",
        )

        self.assertEqual(("delete_agent",), env.model_admissible_action_types())
        self.assertEqual(
            ["failed_reader"],
            env.model_admissible_action_targets()["delete_agent"]["agent_ids"],
        )
        recovery = env.recovery_state()
        self.assertEqual(["delete_agent"], recovery["preferred_actions"])
        self.assertEqual(
            ["failed_reader"],
            recovery[
                "repair_exhausted_auxiliary_takeover_delete_agent_ids"
            ],
        )
        rejected_add = await env.step(
            json.dumps(
                {
                    "action": "add_subgraph",
                    "agents": [
                        {
                            "agent_id": "another_reader",
                            "model_id": "cheap",
                            "contract": "retrieve more evidence",
                            "role_family": "evidence_retriever",
                            "allowed_tools": [QA_RETRIEVAL_TOOL_ID],
                            "execution_mode": "react",
                        }
                    ],
                    "relations": [],
                }
            )
        )
        self.assertFalse(rejected_add.accepted)
        self.assertIn("admissible_delete_agent_ids", rejected_add.feedback)

        deleted = await env.step(
            '{"action":"delete_agent","agent_id":"failed_reader"}'
        )

        self.assertTrue(deleted.accepted)
        self.assertFalse(env.graph.has_node("failed_reader"))
        self.assertTrue(
            env.graph.validate(registry, require_complete=True).valid
        )
        execution = await env.execute()
        self.assertIn("reasoner", execution.outputs)

    async def test_failed_auxiliary_without_successful_takeover_still_augments(
        self,
    ) -> None:
        graph = _hotpot_semantic_graph()
        graph.add_agent(
            AgentNode(
                "failed_reader",
                "cheap",
                "retrieve replacement evidence",
                role_family="evidence_retriever",
                allowed_tools=(QA_RETRIEVAL_TOOL_ID,),
                execution_mode="react",
                artifact_type="retrieval_evidence",
            )
        )
        graph.set_relation("failed_reader", "reasoner", True, False)
        registry = make_registry()
        env = AgentWorkflowEnv(
            registry,
            runtime=_hotpot_semantic_runtime(registry, _ImmediateGateway()),
            graph=graph,
            problem="What is the capital of France?",
            execute_on_edit=False,
            semantic_protocol="hotpotqa_verified_answer_slot_v1",
            recovery_policy="preserve_diagnose_repair_augment",
            required_evidence_tool_id=QA_RETRIEVAL_TOOL_ID,
        )
        env._failed_agent_ids.update({"failed_reader", "reasoner"})
        env._repair_exhausted_agent_ids.update({"failed_reader", "reasoner"})
        env._diagnosed_unusable_agent_ids.add("failed_reader")
        env._latest_failure_record_by_agent["failed_reader"] = AgentFailureRecord(
            request_id="failed-reader-no-takeover",
            agent_id="failed_reader",
            phase=ExecutionPhase.SINGLE,
            graph_revision=graph.revision,
            error_type="ReactExecutionError",
            message="react agent 'failed_reader' exhausted 8 turns",
        )

        self.assertEqual(("add_subgraph",), env.model_admissible_action_types())
        protected = await env.step(
            '{"action":"delete_agent","agent_id":"failed_reader"}'
        )
        self.assertFalse(protected.accepted)
        self.assertTrue(env.graph.has_node("failed_reader"))

    async def test_ordinary_auxiliary_failure_cannot_use_replacement_delete(
        self,
    ) -> None:
        graph = _hotpot_semantic_graph()
        graph.add_agent(
            AgentNode(
                "failed_reader",
                "cheap",
                "retrieve replacement evidence",
                role_family="evidence_retriever",
                allowed_tools=(QA_RETRIEVAL_TOOL_ID,),
                execution_mode="react",
                artifact_type="retrieval_evidence",
            )
        )
        graph.set_relation("failed_reader", "reasoner", True, False)
        registry = make_registry()
        env = AgentWorkflowEnv(
            registry,
            runtime=_hotpot_semantic_runtime(registry, _ImmediateGateway()),
            graph=graph,
            problem="What is the capital of France?",
            execute_on_edit=False,
            semantic_protocol="hotpotqa_verified_answer_slot_v1",
            recovery_policy="preserve_diagnose_repair_augment",
            required_evidence_tool_id=QA_RETRIEVAL_TOOL_ID,
        )
        env._progressive_outputs["reader"] = "grounded replacement evidence"
        env._record_failure_state(
            (
                AgentFailureRecord(
                    request_id="failed-reader-first-attempt",
                    agent_id="failed_reader",
                    phase=ExecutionPhase.SINGLE,
                    graph_revision=graph.revision,
                    error_type="ReactExecutionError",
                    message="react agent 'failed_reader' exhausted 8 turns",
                ),
            ),
            current_agent_ids={node.id for node in graph.nodes},
        )

        self.assertEqual(("modify_agent",), env.model_admissible_action_types())
        protected = await env.step(
            '{"action":"delete_agent","agent_id":"failed_reader"}'
        )
        self.assertFalse(protected.accepted)
        self.assertTrue(env.graph.has_node("failed_reader"))

    async def test_failed_auxiliary_ingress_relation_domain_is_exact_and_complete(
        self,
    ) -> None:
        nodes = [
            AgentNode(
                "successful_repair",
                "cheap",
                "preserve grounded repair evidence",
                role_family="repair",
                artifact_type="repair_evidence",
            ),
            AgentNode(
                "failed_reader",
                "cheap",
                "retrieve replacement evidence",
                role_family="evidence_retriever",
                allowed_tools=(QA_RETRIEVAL_TOOL_ID,),
                execution_mode="react",
                artifact_type="retrieval_evidence",
            ),
            *[
                node
                for node in _hotpot_semantic_graph().nodes
                if node.id != "reader"
            ],
        ]
        graph = AgentGraph(
            nodes,
            [
                AgentRelation("failed_reader", "successful_repair", True, False),
                AgentRelation("failed_reader", "reasoner", True, False),
                AgentRelation("successful_repair", "reasoner", True, False),
                AgentRelation("reasoner", "verifier", True, False),
                AgentRelation("verifier", "formatter", True, False),
            ],
            output_agent_id="formatter",
        )
        registry = make_registry()
        env = AgentWorkflowEnv(
            registry,
            runtime=_hotpot_semantic_runtime(registry, _ImmediateGateway()),
            graph=graph,
            problem="What is the capital of France?",
            execute_on_edit=False,
            semantic_protocol="hotpotqa_verified_answer_slot_v1",
            recovery_policy="preserve_diagnose_repair_augment",
            required_evidence_tool_id=QA_RETRIEVAL_TOOL_ID,
        )
        env._progressive_outputs["successful_repair"] = "grounded repair evidence"
        env._failed_agent_ids.update({"failed_reader", "reasoner"})
        env._repair_exhausted_agent_ids.update({"failed_reader", "reasoner"})

        self.assertEqual(("set_relation",), env.model_admissible_action_types())
        self.assertEqual(
            [
                {
                    "source_id": "failed_reader",
                    "target_id": "reasoner",
                    "source_to_target": False,
                    "target_to_source": False,
                }
            ],
            env.model_admissible_action_targets()["set_relation"]["candidates"],
        )
        rejected_add = await env.step(
            json.dumps(
                {
                    "action": "add_subgraph",
                    "agents": [
                        {
                            "agent_id": "extra_repair",
                            "model_id": "cheap",
                            "contract": "repair evidence",
                            "role_family": "repair",
                            "allowed_tools": [],
                            "execution_mode": "reasoning",
                        }
                    ],
                    "relations": [],
                }
            )
        )
        self.assertFalse(rejected_add.accepted)
        self.assertIn("exact admitted set_relation", rejected_add.feedback)

    async def test_detached_failed_auxiliary_ingress_is_never_routed_back(
        self,
    ) -> None:
        nodes = [
            AgentNode(
                "successful_repair",
                "cheap",
                "preserve grounded repair evidence",
                role_family="repair",
                artifact_type="repair_evidence",
            ),
            AgentNode(
                "failed_reader",
                "cheap",
                "retrieve replacement evidence",
                role_family="evidence_retriever",
                allowed_tools=(QA_RETRIEVAL_TOOL_ID,),
                execution_mode="react",
                artifact_type="retrieval_evidence",
            ),
            *[
                node
                for node in _hotpot_semantic_graph().nodes
                if node.id != "reader"
            ],
        ]
        graph = AgentGraph(
            nodes,
            [
                AgentRelation("failed_reader", "successful_repair", True, False),
                AgentRelation("successful_repair", "reasoner", True, False),
                AgentRelation("reasoner", "verifier", True, False),
                AgentRelation("verifier", "formatter", True, False),
            ],
            output_agent_id="formatter",
        )
        registry = make_registry()
        env = AgentWorkflowEnv(
            registry,
            runtime=_hotpot_semantic_runtime(registry, _ImmediateGateway()),
            graph=graph,
            problem="What is the capital of France?",
            execute_on_edit=False,
            semantic_protocol="hotpotqa_verified_answer_slot_v1",
            recovery_policy="preserve_diagnose_repair_augment",
            required_evidence_tool_id=QA_RETRIEVAL_TOOL_ID,
        )
        env._progressive_outputs["successful_repair"] = "grounded repair evidence"
        env._failed_agent_ids.update({"failed_reader", "reasoner"})
        env._repair_exhausted_agent_ids.update({"failed_reader", "reasoner"})

        reintroduced = {
            "source_id": "failed_reader",
            "target_id": "reasoner",
            "source_to_target": True,
            "target_to_source": False,
        }

        self.assertTrue(
            env._relation_reintroduces_failed_auxiliary_ingress(reintroduced)
        )
        self.assertNotIn(
            reintroduced,
            env._repair_exhausted_relation_candidates(),
        )
        self.assertNotIn(
            reintroduced,
            env._model_admissible_relation_candidates(),
        )
        rejected = await env.step(
            json.dumps({"action": "set_relation", **reintroduced})
        )
        self.assertFalse(rejected.accepted)
        self.assertIn("do not reintroduce", rejected.feedback)

    async def test_terminal_reachability_relation_domain_requires_strict_progress(
        self,
    ) -> None:
        graph = _hotpot_semantic_graph()
        graph.add_agent(
            AgentNode(
                "reachable_repair",
                "cheap",
                "reachable repair evidence",
                role_family="repair",
            )
        )
        graph.set_relation("reachable_repair", "reasoner", True, False)
        graph.add_agent(
            AgentNode(
                "orphan_repair",
                "cheap",
                "orphan repair evidence",
                role_family="repair",
            )
        )
        registry = make_registry()
        env = AgentWorkflowEnv(
            registry,
            runtime=_hotpot_semantic_runtime(registry, _ImmediateGateway()),
            graph=graph,
            problem="What is the capital of France?",
            execute_on_edit=False,
            semantic_protocol="hotpotqa_verified_answer_slot_v1",
            recovery_policy="preserve_diagnose_repair_augment",
            required_evidence_tool_id=QA_RETRIEVAL_TOOL_ID,
        )

        before = set(env._terminal_unreachable_agent_ids())
        self.assertEqual({"orphan_repair"}, before)
        targets = env.model_admissible_action_targets()
        self.assertEqual(("set_relation",), env.model_admissible_action_types())
        self.assertTrue(targets["set_relation"]["candidates"])
        for item in targets["set_relation"]["candidates"]:
            candidate = env.graph.fork()
            candidate.set_relation(
                str(item["source_id"]),
                str(item["target_id"]),
                bool(item["source_to_target"]),
                bool(item["target_to_source"]),
            )
            after = {
                agent_id
                for issue in candidate.validate(
                    registry,
                    require_complete=True,
                ).issues
                if issue.code == "cannot_reach_output"
                for agent_id in issue.agent_ids
            }
            self.assertLess(after, before)

        unrelated = await env.step(
            '{"action":"set_relation","source_id":"reader",'
            '"target_id":"reachable_repair","source_to_target":true,'
            '"target_to_source":false}'
        )
        self.assertFalse(unrelated.accepted)
        self.assertIn("strictly reduces terminal_unreachable_agent_ids", unrelated.feedback)

    async def test_failed_reader_replacement_domain_excludes_cross_role_artifact(
        self,
    ) -> None:
        graph = _hotpot_semantic_graph()
        graph.add_agent(
            AgentNode(
                "failed_reader",
                "cheap",
                "retrieve replacement evidence",
                role_family="evidence_retriever",
                allowed_tools=(QA_RETRIEVAL_TOOL_ID,),
                execution_mode="react",
                artifact_type="retrieval_evidence",
            )
        )
        graph.set_relation("failed_reader", "reasoner", True, False)
        graph.add_agent(
            AgentNode(
                "successful_repair",
                "cheap",
                "repair evidence",
                role_family="repair",
                artifact_type="repair_evidence",
            )
        )
        graph.set_relation("successful_repair", "reasoner", True, False)
        registry = make_registry()
        env = AgentWorkflowEnv(
            registry,
            runtime=_hotpot_semantic_runtime(registry, _ImmediateGateway()),
            graph=graph,
            problem="What is the capital of France?",
            execute_on_edit=False,
            semantic_protocol="hotpotqa_verified_answer_slot_v1",
            recovery_policy="preserve_diagnose_repair_augment",
            required_evidence_tool_id=QA_RETRIEVAL_TOOL_ID,
        )
        env._progressive_outputs["successful_repair"] = "successful repair artifact"
        env._failed_agent_ids.update({"failed_reader", "reasoner"})
        env._repair_exhausted_agent_ids.update({"failed_reader", "reasoner"})
        env._latest_failure_record_by_agent["failed_reader"] = AgentFailureRecord(
            request_id="failed-reader-role-domain",
            agent_id="failed_reader",
            phase=ExecutionPhase.SINGLE,
            graph_revision=graph.revision,
            error_type="ReactExecutionError",
            message="react agent 'failed_reader' exhausted 8 turns",
        )

        self.assertEqual(("add_subgraph",), env.model_admissible_action_types())
        add_domain = env.model_admissible_action_targets()["add_subgraph"]
        self.assertEqual(
            ["evidence_retriever"],
            add_domain["admitted_new_role_families"],
        )
        self.assertEqual(
            ["retrieval_evidence"],
            add_domain["role_constraints"]["evidence_retriever"][
                "artifact_types"
            ],
        )
        self.assertIn("artifact_type", add_domain["required_agent_fields"])
        self.assertNotIn("delete_agent", env.model_admissible_action_types())

        wrong_role = await env.step(
            json.dumps(
                {
                    "action": "add_subgraph",
                    "agents": [
                        {
                            "agent_id": "another_repair",
                            "model_id": "cheap",
                            "contract": "another repair",
                            "role_family": "repair",
                            "allowed_tools": [],
                            "execution_mode": "reasoning",
                            "artifact_type": "repair_evidence",
                        }
                    ],
                    "relations": [],
                }
            )
        )
        self.assertFalse(wrong_role.accepted)
        self.assertIn("same-role/same-artifact replacement", wrong_role.feedback)

    def test_successful_same_role_downstream_takeover_exposes_only_delete(
        self,
    ) -> None:
        graph = _hotpot_semantic_graph()
        for agent_id in ("failed_reader", "replacement_reader"):
            graph.add_agent(
                AgentNode(
                    agent_id,
                    "cheap",
                    "retrieve replacement evidence",
                    role_family="evidence_retriever",
                    allowed_tools=(QA_RETRIEVAL_TOOL_ID,),
                    execution_mode="react",
                    artifact_type="retrieval_evidence",
                )
            )
            graph.set_relation(agent_id, "reasoner", True, False)
        registry = make_registry()
        env = AgentWorkflowEnv(
            registry,
            runtime=_hotpot_semantic_runtime(registry, _ImmediateGateway()),
            graph=graph,
            problem="What is the capital of France?",
            execute_on_edit=False,
            semantic_protocol="hotpotqa_verified_answer_slot_v1",
            recovery_policy="preserve_diagnose_repair_augment",
            required_evidence_tool_id=QA_RETRIEVAL_TOOL_ID,
        )
        env._progressive_outputs["replacement_reader"] = "replacement evidence"
        env._failed_agent_ids.update({"failed_reader", "reasoner"})
        env._repair_exhausted_agent_ids.update({"failed_reader", "reasoner"})
        env._diagnosed_unusable_agent_ids.add("failed_reader")
        env._latest_failure_record_by_agent["failed_reader"] = AgentFailureRecord(
            request_id="failed-reader-takeover",
            agent_id="failed_reader",
            phase=ExecutionPhase.SINGLE,
            graph_revision=graph.revision,
            error_type="ReactExecutionError",
            message="react agent 'failed_reader' exhausted 8 turns",
        )

        self.assertEqual(("delete_agent",), env.model_admissible_action_types())
        self.assertEqual(
            ["failed_reader"],
            env.model_admissible_action_targets()["delete_agent"]["agent_ids"],
        )

    def test_tc9_shape_never_toggles_detached_failed_ingress(self) -> None:
        graph = _hotpot_semantic_graph()
        graph.add_agent(
            AgentNode(
                "failed_reader",
                "cheap",
                "retrieve evidence",
                role_family="evidence_retriever",
                allowed_tools=(QA_RETRIEVAL_TOOL_ID,),
                execution_mode="react",
            )
        )
        graph.add_agent(
            AgentNode(
                "successful_repair",
                "cheap",
                "repair evidence",
                role_family="repair",
            )
        )
        graph.set_relation("failed_reader", "successful_repair", True, False)
        graph.set_relation("failed_reader", "reasoner", True, False)
        graph.set_relation("successful_repair", "reasoner", True, False)
        registry = make_registry()
        env = AgentWorkflowEnv(
            registry,
            runtime=_hotpot_semantic_runtime(registry, _ImmediateGateway()),
            graph=graph,
            problem="What is the capital of France?",
            execute_on_edit=False,
            max_agents=6,
            semantic_protocol="hotpotqa_verified_answer_slot_v1",
            recovery_policy="preserve_diagnose_repair_augment",
            required_evidence_tool_id=QA_RETRIEVAL_TOOL_ID,
        )
        env._progressive_outputs["successful_repair"] = "repair artifact"
        env._failed_agent_ids.update({"failed_reader", "reasoner"})
        env._repair_exhausted_agent_ids.update({"failed_reader", "reasoner"})

        detach_domain = env.model_admissible_action_targets()
        self.assertEqual(
            [
                {
                    "source_id": "failed_reader",
                    "target_id": "reasoner",
                    "source_to_target": False,
                    "target_to_source": False,
                }
            ],
            detach_domain["set_relation"]["candidates"],
        )
        env._graph.set_relation("failed_reader", "reasoner", False, False)

        next_domain = env.model_admissible_action_targets()
        for candidate in next_domain.get("set_relation", {}).get("candidates", []):
            self.assertFalse(
                candidate["source_id"] == "failed_reader"
                and candidate["target_id"] == "reasoner"
                and candidate["source_to_target"] is True
            )

        # Even when the failed auxiliary becomes terminal-unreachable, the
        # reachability repair domain must not bypass the failed-ingress guard.
        env._graph.set_relation("failed_reader", "successful_repair", False, False)
        self.assertIn("failed_reader", env._terminal_unreachable_agent_ids())
        for candidate in env._terminal_reachability_relation_candidates():
            self.assertFalse(
                env._relation_reintroduces_failed_auxiliary_ingress(candidate)
            )
            self.assertFalse(
                candidate["source_id"] == "failed_reader"
                and candidate["target_id"] == "reasoner"
                and candidate["source_to_target"] is True
            )

    def test_generic_modify_domain_excludes_repair_exhausted_agent(self) -> None:
        graph = _hotpot_semantic_graph()
        graph.add_agent(
            AgentNode(
                "failed_reader",
                "cheap",
                "retrieve replacement evidence",
                role_family="evidence_retriever",
                allowed_tools=(QA_RETRIEVAL_TOOL_ID,),
                execution_mode="react",
                artifact_type="retrieval_evidence",
            )
        )
        graph.set_relation("failed_reader", "reasoner", True, False)
        registry = make_registry()
        env = AgentWorkflowEnv(
            registry,
            runtime=_hotpot_semantic_runtime(registry, _ImmediateGateway()),
            graph=graph,
            problem="What is the capital of France?",
            execute_on_edit=False,
            semantic_protocol="hotpotqa_verified_answer_slot_v1",
            recovery_policy="preserve_diagnose_repair_augment",
            required_evidence_tool_id=QA_RETRIEVAL_TOOL_ID,
        )
        env._failed_agent_ids.add("failed_reader")
        env._repair_exhausted_agent_ids.add("failed_reader")
        env._unresolved_dirty_agents.update({"failed_reader", "formatter"})

        self.assertNotIn(
            "failed_reader",
            env._model_admissible_modify_agent_ids(),
        )

    def test_agent_limit_reopens_repair_exhausted_auxiliary(self) -> None:
        graph = _hotpot_semantic_graph()
        graph.add_agent(
            AgentNode(
                "failed_reader",
                "cheap",
                "retrieve replacement evidence",
                role_family="evidence_retriever",
                allowed_tools=(QA_RETRIEVAL_TOOL_ID,),
                execution_mode="react",
                artifact_type="retrieval_evidence",
            )
        )
        graph.set_relation("failed_reader", "reasoner", True, False)
        registry = make_registry()
        env = AgentWorkflowEnv(
            registry,
            runtime=_hotpot_semantic_runtime(registry, _ImmediateGateway()),
            graph=graph,
            problem="What is the capital of France?",
            execute_on_edit=False,
            max_agents=len(graph.nodes),
            semantic_protocol="hotpotqa_verified_answer_slot_v1",
            recovery_policy="preserve_diagnose_repair_augment",
            required_evidence_tool_id=QA_RETRIEVAL_TOOL_ID,
        )
        env._failed_agent_ids.add("failed_reader")
        env._react_exhausted_agent_ids.add("failed_reader")
        env._repair_exhausted_agent_ids.add("failed_reader")
        env._unresolved_dirty_agents.add("failed_reader")

        self.assertEqual(
            ("failed_reader",),
            env._mandatory_repair_agent_ids(),
        )
        self.assertEqual(("modify_agent",), env.model_admissible_action_types())
        self.assertEqual(
            ["failed_reader"],
            env.model_admissible_action_targets()["modify_agent"]["agent_ids"],
        )
        self.assertEqual(
            ["modify_agent"],
            env.recovery_state()["preferred_actions"],
        )

    async def test_max_agents_dirty_replacement_excludes_downstream_modify(
        self,
    ) -> None:
        graph = _hotpot_semantic_graph()
        for agent_id in ("failed_reader", "replacement_reader"):
            graph.add_agent(
                AgentNode(
                    agent_id,
                    "cheap",
                    "retrieve replacement evidence",
                    role_family="evidence_retriever",
                    allowed_tools=(QA_RETRIEVAL_TOOL_ID,),
                    execution_mode="react",
                    artifact_type="retrieval_evidence",
                )
            )
        graph.set_relation("failed_reader", "reasoner", True, False)
        registry = make_registry()
        gateway = _ImmediateGateway()
        env = AgentWorkflowEnv(
            registry,
            runtime=_hotpot_semantic_runtime(registry, gateway),
            graph=graph,
            problem="What is the capital of France?",
            execute_on_edit=False,
            max_agents=6,
            semantic_protocol="hotpotqa_verified_answer_slot_v1",
            recovery_policy="preserve_diagnose_repair_augment",
            required_evidence_tool_id=QA_RETRIEVAL_TOOL_ID,
        )
        env._failed_agent_ids.update({"failed_reader", "reasoner"})
        env._repair_exhausted_agent_ids.update({"failed_reader", "reasoner"})
        env._unresolved_dirty_agents.update(
            {"replacement_reader", "reasoner", "verifier", "formatter"}
        )

        self.assertEqual(("modify_agent",), env.model_admissible_action_types())
        modify_domain = env.model_admissible_action_targets()["modify_agent"]
        self.assertEqual(
            ["replacement_reader"],
            modify_domain["agent_ids"],
        )
        self.assertEqual(
            ["contract", "completion_condition"],
            modify_domain["mutable_fields"],
        )
        self.assertEqual(
            ["contract", "completion_condition"],
            modify_domain["per_agent_candidates"][0]["mutable_fields"],
        )
        original_artifact_type = env.graph.get_node(
            "replacement_reader"
        ).artifact_type
        rejected_artifact = await env.step(
            '{"action":"modify_agent","agent_id":"replacement_reader",'
            '"artifact_type":"repair_evidence"}'
        )
        self.assertFalse(rejected_artifact.accepted)
        self.assertIn(
            "mutable_fields=['contract', 'completion_condition']",
            rejected_artifact.feedback,
        )
        self.assertEqual(
            original_artifact_type,
            env.graph.get_node("replacement_reader").artifact_type,
        )
        rejected = await env.step(
            '{"action":"modify_agent","agent_id":"verifier",'
            '"contract":"verify the replacement artifact"}'
        )
        self.assertFalse(rejected.accepted)
        self.assertIn("before modifying blocked downstream Agents", rejected.feedback)
        self.assertEqual([], gateway.requests)
        accepted = await env.step(
            '{"action":"modify_agent","agent_id":"replacement_reader",'
            '"contract":"retry retrieval while preserving the artifact type"}'
        )
        self.assertTrue(accepted.accepted, accepted.feedback)
        self.assertEqual(
            "retry retrieval while preserving the artifact type",
            env.graph.get_node("replacement_reader").contract,
        )

    async def test_tc10_reader_replacement_continues_public_tool_state_then_deletes(
        self,
    ) -> None:
        graph = _hotpot_semantic_graph()
        graph.add_agent(
            AgentNode(
                "failed_reader",
                "cheap",
                "retrieve evidence with the public QA Tool",
                role_family="evidence_retriever",
                allowed_tools=(QA_RETRIEVAL_TOOL_ID,),
                execution_mode="react",
                artifact_type="retrieval_evidence",
            )
        )
        graph.set_relation("failed_reader", "reasoner", True, False)
        registry = make_registry()
        runtime = _hotpot_semantic_runtime(registry, _ImmediateGateway())
        env = AgentWorkflowEnv(
            registry,
            runtime=runtime,
            graph=graph,
            problem="What is the capital of France?",
            execute_on_edit=True,
            semantic_protocol="hotpotqa_verified_answer_slot_v1",
            recovery_policy="preserve_diagnose_repair_augment",
            required_evidence_tool_id=QA_RETRIEVAL_TOOL_ID,
        )
        tool_trace = {
            "turn": 1,
            "structured_action": {
                "kind": "tool",
                "name": "read",
                "resource_id": QA_RETRIEVAL_TOOL_ID,
                "arguments": {"passage_id": "tc10-public"},
            },
            "observation": {
                "observation_status": "success",
                "executed_action": {
                    "kind": "tool",
                    "name": "read",
                    "resource_id": QA_RETRIEVAL_TOOL_ID,
                    "arguments": {"passage_id": "tc10-public"},
                },
            },
        }
        rejected_completion = {
            "turn": 2,
            "observation": {
                "observation_status": "schema_invalid",
                "executed_action": {
                    "kind": "complete",
                    "name": "complete",
                    "resource_id": None,
                    "arguments": {"value": "role-specific completion"},
                },
            },
        }
        receipt = _test_read_receipt("tc10-public")
        failure = AgentFailureRecord(
            request_id="tc10-reader-exhausted",
            agent_id="failed_reader",
            phase=ExecutionPhase.SINGLE,
            graph_revision=graph.revision,
            error_type="ReactExecutionError",
            message="react agent 'failed_reader' exhausted 8 turns",
            metadata={
                "react_trace": [tool_trace, rejected_completion],
                "tool_receipts": [receipt],
                "node_unusable": True,
            },
        )
        env._record_failure_state(
            (failure,),
            current_agent_ids={node.id for node in graph.nodes},
        )
        env._repair_exhausted_agent_ids.add("failed_reader")
        env._failed_agent_ids.add("reasoner")
        env._repair_exhausted_agent_ids.add("reasoner")

        replacement_domain = env.model_admissible_action_targets()[
            "add_subgraph"
        ]
        self.assertEqual([], replacement_domain["relations"])
        self.assertIsNone(replacement_domain["output_agent_id"])
        self.assertEqual(
            ["retrieval_evidence"],
            replacement_domain["role_constraints"]["evidence_retriever"][
                "artifact_types"
            ],
        )

        action_payload = json.dumps(
            {
                "action": "add_subgraph",
                "agents": [
                    {
                        "agent_id": "replacement_reader",
                        "model_id": "cheap",
                        "contract": "continue public grounded evidence retrieval",
                        "role_family": "evidence_retriever",
                        "allowed_tools": [QA_RETRIEVAL_TOOL_ID],
                        "execution_mode": "react",
                        "artifact_type": "retrieval_evidence",
                    }
                ],
                "relations": [],
            }
        )
        observed_failure_metadata: dict[str, object] = {}

        async def replacement_success(
            candidate_graph: AgentGraph,
            *args: object,
            **kwargs: object,
        ) -> AgentRuntimeResult:
            del args
            observed_failure_metadata.update(kwargs["prior_failure_metadata"])
            return AgentRuntimeResult(
                run_id="tc10-replacement",
                graph_revision=candidate_graph.revision,
                output_agent_id=candidate_graph.output_agent_id,
                final_answer=None,
                outputs={"replacement_reader": "replacement evidence"},
                output_metadata={
                    "replacement_reader": {
                        "continuation_source_agent_id": "failed_reader",
                        "tool_receipts": [receipt],
                    }
                },
                calls=(),
                block_completion_order=(("replacement_reader",),),
                executed_agent_ids=("replacement_reader",),
                deferred_agent_ids=("reasoner", "verifier", "formatter"),
            )

        inline_payload = json.loads(action_payload)
        inline_payload["relations"] = [
            {
                "source_id": "replacement_reader",
                "target_id": "reasoner",
                "source_to_target": True,
                "target_to_source": False,
            }
        ]
        rejected_inline = await env.step(json.dumps(inline_payload))
        self.assertFalse(rejected_inline.accepted)
        self.assertIn("isolated executable prefix", rejected_inline.feedback)

        self.assertIn("add_subgraph", env.model_admissible_action_types())
        with patch.object(runtime, "execute", side_effect=replacement_success):
            result = await env.step(action_payload)

        self.assertTrue(result.accepted, result.feedback)
        projected = observed_failure_metadata["replacement_reader"]
        self.assertEqual(
            "failed_reader",
            projected["continuation_source_agent_id"],
        )
        self.assertEqual([tool_trace], projected["react_trace"])
        self.assertEqual([receipt], projected["tool_receipts"])
        self.assertNotIn("private_reasoning", projected)
        self.assertEqual(
            (),
            env.graph.directed_predecessors("replacement_reader"),
        )
        self.assertEqual(
            (),
            env._directed_successors(env.graph, "replacement_reader"),
        )
        self.assertEqual(("set_relation",), env.model_admissible_action_types())
        route_candidates = env.model_admissible_action_targets()["set_relation"][
            "candidates"
        ]
        self.assertEqual(
            [
                {
                    "source_id": "replacement_reader",
                    "target_id": "reasoner",
                    "source_to_target": True,
                    "target_to_source": False,
                }
            ],
            route_candidates,
        )
        with patch.object(runtime, "execute", side_effect=replacement_success):
            routed = await env.step(
                json.dumps({"action": "set_relation", **route_candidates[0]})
            )
        self.assertTrue(routed.accepted, routed.feedback)
        self.assertEqual(
            ("reasoner",),
            env._directed_successors(env.graph, "replacement_reader"),
        )
        recovery = env.recovery_state()
        self.assertIn("failed_reader", recovery["deletable_agent_ids"])
        self.assertIn(
            "failed_reader",
            recovery["repair_exhausted_auxiliary_takeover_delete_agent_ids"],
        )
        self.assertEqual(("delete_agent",), env.model_admissible_action_types())
        self.assertEqual(
            ["failed_reader"],
            env.model_admissible_action_targets()["delete_agent"]["agent_ids"],
        )

    def test_scheduler_cancellation_preserves_receipt_without_failure_state(
        self,
    ) -> None:
        graph = _hotpot_semantic_graph()
        registry = make_registry()
        env = AgentWorkflowEnv(
            registry,
            runtime=_hotpot_semantic_runtime(registry, _ImmediateGateway()),
            graph=graph,
            problem="What is the capital of France?",
            execute_on_edit=False,
            semantic_protocol="hotpotqa_verified_answer_slot_v1",
            recovery_policy="preserve_diagnose_repair_augment",
            required_evidence_tool_id=QA_RETRIEVAL_TOOL_ID,
        )
        receipt = _test_read_receipt("cancelled-prefix")
        record = AgentFailureRecord(
            request_id="scheduler-cancelled-reader",
            agent_id="reader",
            phase=ExecutionPhase.SINGLE,
            graph_revision=graph.revision,
            error_type="CancelledError",
            message="Agent invocation was cancelled after partial public execution",
            metadata={
                "react_trace": [
                    {
                        "turn": 1,
                        "observation": {
                            "executed_action": {
                                "kind": "tool",
                                "resource_id": QA_RETRIEVAL_TOOL_ID,
                            }
                        },
                    }
                ],
                "tool_receipts": [receipt],
            },
        )

        env._record_failure_state(
            (record,),
            current_agent_ids={node.id for node in graph.nodes},
        )

        self.assertEqual(
            "sibling_fail_fast_cancellation",
            env._execution_failure_diagnosis(record)[0],
        )
        self.assertNotIn("reader", env._failed_agent_ids)
        self.assertNotIn("reader", env._latest_failure_record_by_agent)
        self.assertNotIn("reader", env._react_exhausted_agent_ids)
        self.assertNotIn("reader", env._repair_exhausted_agent_ids)
        self.assertEqual((), env._mandatory_repair_agent_ids())
        self.assertEqual(
            [receipt],
            env._failure_continuations["reader"]["tool_receipts"],
        )

    async def test_hotpot_cached_semantic_diagnostic_cannot_replace_structure_target(
        self,
    ) -> None:
        registry = make_registry()
        gateway = _HotpotSemanticGateway(
            reasoner_candidate="Lyon",
            verifier_candidate="Paris",
        )
        env = AgentWorkflowEnv(
            registry,
            runtime=_hotpot_semantic_runtime(registry, gateway),
            graph=_hotpot_semantic_graph(),
            problem="What is the capital of France?",
            semantic_protocol="hotpotqa_verified_answer_slot_v1",
            recovery_policy="preserve_diagnose_repair_augment",
            required_evidence_tool_id="qa-retrieval",
        )
        execution = await env.execute()
        env._progressive_execution = execution
        env._graph.add_agent(
            AgentNode(
                "orphan",
                "cheap",
                "supplementary diagnosis",
                role_family="repair",
            )
        )
        env._progressive_execution_revision = env.graph.revision

        admission = env.finish_admissibility()

        self.assertEqual("graph_validation", admission["stage"])
        self.assertIn("semantic_lineage_diagnostic", admission)
        self.assertEqual(
            "terminal_reachability",
            admission["failure_attribution"]["responsible_constraint"],
        )
        self.assertEqual(
            ["orphan"],
            admission["failure_attribution"]["responsible_agent_ids"],
        )

    async def test_model_admissible_mask_excludes_empty_relation_domain(self) -> None:
        env = AgentWorkflowEnv(
            make_registry(),
            _ImmediateGateway(),
            graph=AgentGraph(
                [
                    AgentNode("source", "cheap", "source"),
                    AgentNode("output", "fast", "output"),
                ],
                output_agent_id="output",
            ),
            problem="question",
        )

        with patch.object(
            env,
            "_model_admissible_relation_candidates",
            return_value=[],
        ):
            self.assertNotIn(
                "set_relation",
                env.model_admissible_action_types(),
            )

    async def test_hotpot_semantic_protocol_accepts_only_verified_answer_lineage(self) -> None:
        registry = make_registry()
        gateway = _HotpotSemanticGateway()
        env = AgentWorkflowEnv(
            registry,
            runtime=_hotpot_semantic_runtime(registry, gateway),
            graph=_hotpot_semantic_graph(),
            problem="What is the capital of France?",
            require_exact_answer_tag=True,
            require_format_agent=True,
            execute_on_edit=True,
            semantic_protocol="hotpotqa_verified_answer_slot_v1",
            recovery_policy="preserve_diagnose_repair_augment",
            required_evidence_tool_id="qa-retrieval",
        )

        before_execution = env.finish_admissibility()
        self.assertFalse(before_execution["admissible"])
        self.assertEqual("execution", before_execution["stage"])
        self.assertEqual(
            "output_artifact",
            before_execution["failure_attribution"]["responsible_constraint"],
        )
        self.assertEqual(
            "formatter",
            before_execution["failure_attribution"]["responsible_agent_id"],
        )
        env._failed_agent_ids.add("reasoner")
        failed_execution = env.finish_admissibility()
        self.assertEqual(
            "execution_contract_or_runtime_failure",
            failed_execution["failure_attribution"]["responsible_constraint"],
        )
        self.assertEqual(
            "reasoner",
            failed_execution["failure_attribution"]["responsible_agent_id"],
        )
        self.assertEqual(
            ["reasoner"],
            failed_execution["failure_attribution"]["responsible_agent_ids"],
        )
        env._failed_agent_ids.clear()
        executed = await env.step(
            '{"action":"modify_agent","agent_id":"reader",'
            '"contract":"read explicit database evidence"}'
        )
        self.assertTrue(executed.accepted)
        self.assertTrue(env.finish_admissibility()["admissible"])
        self.assertEqual(("finish",), env.model_admissible_action_types())
        self.assertEqual(
            {"finish": {"admissible": True, "submission_semantics": "explicit_finish"}},
            env.model_admissible_action_targets(),
        )
        protected = await env.step(
            '{"action":"modify_agent","agent_id":"verifier",'
            '"contract":"reconsider the answer"}'
        )
        self.assertFalse(protected.accepted)
        self.assertIn("verified terminal artifact", protected.feedback)
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
            runtime=_hotpot_semantic_runtime(registry, gateway),
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
            runtime=_hotpot_semantic_runtime(registry, gateway),
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
        with self.assertRaisesRegex(
            AgentWorkflowStateError,
            "input only from Reasoners",
        ):
            AgentWorkflowEnv(
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

    async def test_hotpot_formatter_must_wrap_exact_unchanged_candidate(self) -> None:
        registry = make_registry()
        gateway = _HotpotSemanticGateway(formatter_value="Paris, France")
        env = AgentWorkflowEnv(
            registry,
            runtime=_hotpot_semantic_runtime(registry, gateway),
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
            "Alias binding correct: true\n"
            "Answer type cardinality correct: true\n"
            "Multi-hop complete: true\n"
            "Minimal answer surface: true\n"
            "Scope preserved: true\n"
            "Verification status: supported"
        )

        self.assertIsNone(reasoner_candidate)
        self.assertIn("answer_slot", reasoner_issue or "")
        self.assertIsNone(verifier_issue)
        self.assertEqual("Paris", verifier_candidate)
        numeric_candidate, numeric_issue = env._verifier_candidate(
            "Candidate answer: 1844  \n"
            "Evidence supported: true\n"
            "Entity attribute binding correct: true\n"
            "Alias binding correct: true\n"
            "Answer type cardinality correct: true\n"
            "Multi-hop complete: true\n"
            "Minimal answer surface: true\n"
            "Scope preserved: true\n"
            "Verification status: supported"
        )
        self.assertEqual("1844", numeric_candidate)
        self.assertIsNone(numeric_issue)

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
                "answer_type": "short_answer",
                "answer_cardinality": "single",
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

        forged_candidate = json.loads(json.dumps(artifact))
        forged_candidate["answer_slot"]["answer_field"] = "subject"
        forged_candidate["candidate_answer"] = "Chester"
        candidate, issue = env._reasoner_candidate(
            json.dumps(forged_candidate),
            original_question="What is the capital of France?",
        )
        self.assertIsNone(candidate)
        self.assertIn(
            "must occur verbatim in the selected evidence_span",
            str(issue),
        )

        misbound_slot = json.loads(json.dumps(artifact))
        misbound_slot["answer_slot"]["answer_field"] = "subject"
        candidate, issue = env._reasoner_candidate(
            json.dumps(misbound_slot),
            original_question="What is the capital of France?",
        )
        self.assertIsNone(candidate)
        self.assertIn("answer_field selects 'subject'", str(issue))
        self.assertIn(
            "matches the selected proposition field "
            "'object_or_attribute_value'",
            str(issue),
        )

        comparison = json.loads(json.dumps(artifact))
        comparison["question_scope"] = "Which magazine was started first, A or B?"
        comparison["answer_slot"]["answer_type"] = "entity"
        comparison["answer_slot"]["answer_field"] = "object_or_attribute_value"
        comparison["evidence_propositions"][0].update(
            {
                "subject": "A",
                "relation": "publication date",
                "object_or_attribute_value": "1844",
                "evidence_span": "A was first published in 1844.",
            }
        )
        comparison["candidate_answer"] = "1844"
        candidate, issue = env._reasoner_candidate(
            json.dumps(comparison),
            original_question="Which magazine was started first, A or B?",
        )
        self.assertIsNone(candidate)
        self.assertIn("requires answer type 'entity'", str(issue))

        possessive = json.loads(json.dumps(artifact))
        possessive["question_scope"] = "The character was named after who?"
        possessive["answer_slot"]["answer_type"] = "entity"
        possessive["evidence_propositions"][0].update(
            {
                "subject": "Milhouse",
                "relation": "named after",
                "object_or_attribute_value": "President Nixon's middle name",
                "evidence_span": (
                    "Milhouse was named after President Nixon's middle name."
                ),
            }
        )
        possessive["candidate_answer"] = "President Nixon's middle name"
        candidate, issue = env._reasoner_candidate(
            json.dumps(possessive),
            original_question="The character was named after who?",
        )
        self.assertIsNone(candidate)
        self.assertIn("possessive noun phrase", str(issue))

        incomplete_surface = json.loads(json.dumps(artifact))
        incomplete_surface["question_scope"] = "The role was named after who?"
        incomplete_surface["answer_slot"]["answer_type"] = "entity"
        incomplete_surface["evidence_propositions"][0].update(
            {
                "subject": "The role",
                "relation": "named after",
                "object_or_attribute_value": "Ada Lovelace",
                "evidence_span": (
                    "The role was named after Professor Ada Lovelace's surname."
                ),
            }
        )
        incomplete_surface["candidate_answer"] = "Ada Lovelace"
        candidate, issue = env._reasoner_candidate(
            json.dumps(incomplete_surface),
            original_question="The role was named after who?",
        )
        self.assertIsNone(candidate)
        self.assertIn("strict subspan", str(issue))
        self.assertIn("complete referential surface", str(issue))

        complete_surface = json.loads(json.dumps(incomplete_surface))
        complete_surface["evidence_propositions"][0][
            "object_or_attribute_value"
        ] = "Professor Ada Lovelace"
        complete_surface["candidate_answer"] = "Professor Ada Lovelace"
        candidate, issue = env._reasoner_candidate(
            json.dumps(complete_surface),
            original_question="The role was named after who?",
        )
        self.assertEqual("Professor Ada Lovelace", candidate)
        self.assertIsNone(issue)

        unqualified_surface = json.loads(json.dumps(incomplete_surface))
        unqualified_surface["evidence_propositions"][0]["evidence_span"] = (
            "The role was named after Ada Lovelace's surname."
        )
        candidate, issue = env._reasoner_candidate(
            json.dumps(unqualified_surface),
            original_question="The role was named after who?",
        )
        self.assertEqual("Ada Lovelace", candidate)
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
        self.assertIn("must occur verbatim in the selected evidence_span", str(issue))

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

    async def test_neutral_canvas_rejects_react_role_but_admits_react_execution_mode(
        self,
    ) -> None:
        registry = make_registry()
        gateway = _ImmediateGateway()
        runtime = AgentRuntime(
            registry,
            gateway,
            execution_adapters={"react": ReasoningExecutionAdapter(gateway)},
        )
        env = AgentWorkflowEnv(
            registry,
            runtime=runtime,
            problem="question",
            semantic_protocol="none",
        )

        rejected = await env.step(
            '{"action":"add_agent","agent_id":"bad","model_id":"balanced",'
            '"contract":"retrieve","role_family":"ReAct",'
            '"execution_mode":"reasoning"}'
        )
        accepted = await env.step(
            '{"action":"add_agent","agent_id":"reader","model_id":"balanced",'
            '"contract":"retrieve evidence","role_family":"evidence_hunter",'
            '"execution_mode":"react"}'
        )

        self.assertFalse(rejected.accepted)
        self.assertIn("ReAct is an execution_mode", rejected.feedback)
        self.assertTrue(accepted.accepted)
        self.assertEqual("react", env.graph.get_node("reader").execution_mode.value)

    async def test_hotpot_semantic_edits_require_roles_and_exact_reasoner_tool_mode(
        self,
    ) -> None:
        env = AgentWorkflowEnv(
            make_registry(),
            _ImmediateGateway(),
            problem="question",
            semantic_protocol="hotpotqa_verified_answer_slot_v1",
            recovery_policy="preserve_diagnose_repair_augment",
            required_evidence_tool_id="qa-retrieval",
        )
        missing_role = AgentGraph(
            [AgentNode("unknown", "balanced", "untyped contract")]
        )
        wrong_reasoner = AgentGraph(
            [
                AgentNode(
                    "reasoner",
                    "balanced",
                    "align answer slot",
                    role_family="reasoner",
                )
            ]
        )
        exact_reasoner = AgentGraph(
            [
                AgentNode(
                    "reasoner",
                    "balanced",
                    "align answer slot",
                    role_family="reasoner",
                    allowed_tools=(QA_RETRIEVAL_TOOL_ID,),
                    execution_mode="react",
                )
            ]
        )

        self.assertIn(
            "non-empty role_family",
            env._semantic_edit_issue_for(missing_role) or "",
        )
        self.assertIn(
            "exactly allowed_tools=['qa-retrieval']",
            env._semantic_edit_issue_for(wrong_reasoner) or "",
        )
        self.assertIsNone(env._semantic_edit_issue_for(exact_reasoner))

    async def test_hotpot_live_action_targets_filter_relation_candidates(self) -> None:
        env = AgentWorkflowEnv(
            make_registry(),
            _ImmediateGateway(),
            graph=_hotpot_semantic_graph(),
            problem="What is the capital of France?",
            semantic_protocol="hotpotqa_verified_answer_slot_v1",
            recovery_policy="preserve_diagnose_repair_augment",
            required_evidence_tool_id="qa-retrieval",
        )

        targets = env.model_admissible_action_targets()

        add_domain = targets["add_subgraph"]
        self.assertIn("role_family", add_domain["required_agent_fields"])
        self.assertEqual(
            [["qa-retrieval"]],
            add_domain["role_constraints"]["reasoner"]["allowed_tools"],
        )
        self.assertEqual(
            [_HOTPOTQA_FORMAT_CONTRACT],
            add_domain["role_constraints"]["format"]["contracts"],
        )
        modify_domain = targets["modify_agent"]
        self.assertNotIn("allowed_tools", modify_domain["mutable_fields"])
        self.assertIn("model_id", modify_domain["mutable_fields"])
        self.assertEqual(4, len(modify_domain["per_agent_candidates"]))
        formatter_candidates = next(
            item
            for item in modify_domain["per_agent_candidates"]
            if item["agent_id"] == "formatter"
        )
        self.assertNotIn("contract", formatter_candidates["mutable_fields"])
        self.assertEqual(
            sorted(make_registry().model_ids),
            sorted(add_domain["model_ids"]),
        )
        self.assertIn("evidence_retriever", add_domain["role_constraints"])
        relation_candidates = targets["set_relation"]["candidates"]
        self.assertTrue(relation_candidates)
        self.assertFalse(
            any(
                item["source_id"] == "reader"
                and item["target_id"] == "verifier"
                and item["source_to_target"] is True
                for item in relation_candidates
            )
        )
        self.assertFalse(
            any(
                item == {
                    "source_id": "reasoner",
                    "target_id": "verifier",
                    "source_to_target": True,
                    "target_to_source": False,
                }
                for item in relation_candidates
            )
        )

    async def test_hotpot_format_predecessor_must_be_verifier(self) -> None:
        registry = make_registry()
        gateway = _ImmediateGateway()
        graph = AgentGraph(
            [
                AgentNode("reader", "cheap", "evidence", role_family="evidence_retriever"),
                AgentNode("verifier", "balanced", "verify", role_family="verifier"),
                AgentNode(
                    "reasoner",
                    "balanced",
                    "answer",
                    role_family="reasoner",
                    allowed_tools=(QA_RETRIEVAL_TOOL_ID,),
                    execution_mode="react",
                ),
                AgentNode(
                    "formatter",
                    "fast",
                    _HOTPOTQA_FORMAT_CONTRACT,
                    role_family="format",
                ),
            ],
            [
                AgentRelation("reader", "verifier", True, False),
                AgentRelation("verifier", "reasoner", True, False),
                AgentRelation("reasoner", "formatter", True, False),
            ],
            output_agent_id="formatter",
        )
        with self.assertRaisesRegex(
            AgentWorkflowStateError,
            "input only from Verifiers",
        ):
            AgentWorkflowEnv(
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
            '"contract":"answer","role_family":"reasoner",'
            '"allowed_tools":["qa-retrieval"],"execution_mode":"react"},'
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
        self.assertIn("HotpotQA Format Agent", rejected.feedback)
        self.assertIn("execution_mode='reasoning'", rejected.feedback)
        self.assertEqual((), env.graph.nodes)

    async def test_hotpot_formatter_contract_cannot_preselect_answer(self) -> None:
        registry = make_registry()
        gateway = _ImmediateGateway()
        env = AgentWorkflowEnv(
            registry,
            gateway,
            problem="question",
            semantic_protocol="hotpotqa_verified_answer_slot_v1",
            recovery_policy="preserve_diagnose_repair_augment",
            required_evidence_tool_id="qa-retrieval",
        )

        rejected = await env.step(
            '{"action":"add_subgraph","agents":['
            '{"agent_id":"formatter","model_id":"fast",'
            '"contract":"format the final answer as Paris",'
            '"role_family":"format","allowed_tools":[],'
            '"execution_mode":"reasoning"}],"relations":[]}'
        )

        self.assertFalse(rejected.accepted)
        self.assertIn("neutral formatting-only contract", rejected.feedback)
        self.assertEqual((), env.graph.nodes)

    async def test_hotpot_contract_admission_keeps_obligations_answer_free(
        self,
    ) -> None:
        registry = make_registry()
        gateway = _ImmediateGateway()
        problem = (
            "Based on the following passages, answer the question.\n\n"
            "[Professor Ada Lovelace founded the Meridian Archive beside the "
            "river in 1843.]\n\nQuestion: Who founded the Meridian Archive?"
        )
        env = AgentWorkflowEnv(
            registry,
            runtime=_hotpot_semantic_runtime(registry, gateway),
            problem=problem,
            semantic_protocol="hotpotqa_verified_answer_slot_v1",
            recovery_policy="preserve_diagnose_repair_augment",
            required_evidence_tool_id="qa-retrieval",
        )
        revision = env.revision

        copied_candidate = await env.step(
            json.dumps(
                {
                    "action": "add_subgraph",
                    "agents": [
                        {
                            "agent_id": "reasoner",
                            "model_id": "balanced",
                            "contract": "Return Ada Lovelace as the answer",
                            "role_family": "reasoner",
                            "allowed_tools": ["qa-retrieval"],
                            "execution_mode": "react",
                        }
                    ],
                    "relations": [],
                }
            )
        )

        self.assertFalse(copied_candidate.accepted)
        self.assertIn("pre-execution obligations only", copied_candidate.feedback)
        self.assertEqual(revision, env.revision)
        self.assertEqual((), env.graph.nodes)
        self.assertEqual([], gateway.requests)

        obligation_only = await env.step(
            json.dumps(
                {
                    "action": "add_subgraph",
                    "agents": [
                        {
                            "agent_id": "reasoner",
                            "model_id": "balanced",
                            "contract": (
                                "Preserve the Meridian Archive question scope; align "
                                "the requested person answer slot to explicit evidence"
                            ),
                            "role_family": "reasoner",
                            "allowed_tools": ["qa-retrieval"],
                            "execution_mode": "react",
                        }
                    ],
                    "relations": [],
                }
            )
        )

        self.assertTrue(obligation_only.accepted)
        self.assertTrue(env.graph.has_node("reasoner"))

        schema_repair = await env.step(
            json.dumps(
                {
                    "action": "modify_agent",
                    "agent_id": "reasoner",
                    "contract": (
                        "Return exactly one StructuredAction with arguments, kind, "
                        "name, resource_id, and skill_id at the top level"
                    ),
                }
            )
        )
        self.assertTrue(schema_repair.accepted)

        for concrete_tool_contract in (
            "Call search(query='David Soul birthplace', limit=10).",
            "Perform a focused search for 'Dame Judi Dench birthplace'.",
            'Use {"query": "David Soul birthplace"} for retrieval.',
            "Expand retrieval with top-k=25.",
            'Read the Tool receipt with "passage_id": "atlas:123".',
        ):
            revision = env.revision
            rejected_tool_arguments = await env.step(
                json.dumps(
                    {
                        "action": "modify_agent",
                        "agent_id": "reasoner",
                        "contract": concrete_tool_contract,
                    }
                )
            )
            self.assertFalse(rejected_tool_arguments.accepted)
            self.assertIn(
                "without concrete Tool invocation arguments",
                rejected_tool_arguments.feedback,
            )
            self.assertEqual(revision, env.revision)

        neutral_retrieval_responsibility = await env.step(
            json.dumps(
                {
                    "action": "modify_agent",
                    "agent_id": "reasoner",
                    "contract": (
                        "Rewrite the query using the current entity and relation "
                        "evidence, and expand top-k when retrieval recall is insufficient"
                    ),
                }
            )
        )
        self.assertTrue(neutral_retrieval_responsibility.accepted)

        neutral_tool_reference = await env.step(
            json.dumps(
                {
                    "action": "modify_agent",
                    "agent_id": "reasoner",
                    "contract": (
                        "Retrieve evidence using the 'qa-retrieval' tool; bind "
                        "the entity and relation before semantic completion"
                    ),
                }
            )
        )
        self.assertTrue(neutral_tool_reference.accepted)

        for answer_precommit in (
            {
                "action": "modify_agent",
                "agent_id": "reasoner",
                "contract": "Output ONLY the word 'Shirley'.",
            },
            {
                "action": "modify_agent",
                "agent_id": "reasoner",
                "completion_condition": "Return ONLY the word 'Shirley'.",
            },
            {
                "action": "modify_agent",
                "agent_id": "reasoner",
                "contract": (
                    "Resolve entity 'Sinclair' to 'Joseph C. Lincoln' before "
                    "forming the evidence proposition"
                ),
            },
        ):
            revision = env.revision
            rejected_precommit = await env.step(json.dumps(answer_precommit))
            self.assertFalse(rejected_precommit.accepted)
            self.assertIn(
                "pre-execution obligations only",
                rejected_precommit.feedback,
            )
            self.assertEqual(revision, env.revision)

        env._failure_continuations["reasoner"] = {
            "execution_phase": "single",
            "react_trace": [
                {
                    "structured_action": {
                        "kind": "complete",
                        "arguments": {
                            "value": json.dumps(
                                {
                                    "candidate_answer": "Riverport",
                                    "evidence_span": (
                                        "The archive opened in Riverport beside the river."
                                    ),
                                }
                            )
                        },
                    }
                }
            ],
        }
        revision = env.revision
        copied_public_candidate = await env.step(
            json.dumps(
                {
                    "action": "modify_agent",
                    "agent_id": "reasoner",
                    "contract": "Select Riverport as the candidate answer",
                }
            )
        )
        self.assertFalse(copied_public_candidate.accepted)
        self.assertIn(
            "public Tool/Agent observations",
            copied_public_candidate.feedback,
        )
        self.assertEqual(revision, env.revision)

    async def test_hotpot_formatter_contract_cannot_be_mutated_to_an_answer(self) -> None:
        registry = make_registry()
        env = AgentWorkflowEnv(
            registry,
            runtime=_hotpot_semantic_runtime(registry, _HotpotSemanticGateway()),
            graph=_hotpot_semantic_graph(),
            problem="What is the capital of France?",
            semantic_protocol="hotpotqa_verified_answer_slot_v1",
            recovery_policy="preserve_diagnose_repair_augment",
            required_evidence_tool_id="qa-retrieval",
        )
        revision = env.revision

        rejected = await env.step(
            '{"action":"modify_agent","agent_id":"formatter",'
            '"contract":"format the final answer as Lyon"}'
        )

        self.assertFalse(rejected.accepted)
        self.assertIn("neutral formatting-only contract", rejected.feedback)
        self.assertEqual(revision, env.revision)
        self.assertEqual(
            _HOTPOTQA_FORMAT_CONTRACT,
            env.graph.get_node("formatter").contract,
        )

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
        self.assertEqual(
            ["source"],
            recovery_state["diagnosed_unusable_agent_ids"],
        )
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

    async def test_transient_provider_failure_never_admits_delete_after_takeover(
        self,
    ) -> None:
        registry = make_registry()
        graph = AgentGraph(
            [
                AgentNode(
                    "source",
                    "balanced",
                    "produce evidence",
                    role_family="evidence",
                    artifact_type="facts",
                ),
                AgentNode(
                    "replacement",
                    "cheap",
                    "replacement evidence",
                    role_family="evidence",
                    artifact_type="facts",
                ),
                AgentNode("out", "fast", "answer", role_family="output"),
            ],
            [
                AgentRelation("source", "out", True, False),
                AgentRelation("replacement", "out", True, False),
            ],
            output_agent_id="out",
        )
        env = AgentWorkflowEnv(
            registry,
            _ImmediateGateway(),
            graph=graph,
            problem="question",
            recovery_policy="preserve_diagnose_repair_augment",
        )
        env._progressive_outputs["replacement"] = "replacement evidence"
        env._record_failure_state(
            (
                AgentFailureRecord(
                    request_id="request-429",
                    agent_id="source",
                    phase=ExecutionPhase.SINGLE,
                    graph_revision=graph.revision,
                    error_type="OpenAICompatibleGatewayError",
                    message="provider request failed: HTTP 429",
                ),
            ),
            current_agent_ids={"source", "replacement", "out"},
        )

        issue = env._delete_admission_issue("source")

        self.assertIsNotNone(issue)
        self.assertEqual([], env.recovery_state()["diagnosed_unusable_agent_ids"])
        self.assertIn("node has not been diagnosed unusable", issue)

    async def test_hotpot_reasoner_takeover_requires_valid_semantic_artifact(self) -> None:
        graph = _hotpot_semantic_graph()
        graph.add_agent(
            AgentNode(
                "replacement_reasoner",
                "cheap",
                "replacement semantic alignment",
                role_family="reasoner",
                allowed_tools=(QA_RETRIEVAL_TOOL_ID,),
                execution_mode="react",
                artifact_type="semantic_candidate",
            )
        )
        graph.set_relation("replacement_reasoner", "verifier", True, False)
        registry = make_registry()
        gateway = _ImmediateGateway()
        env = AgentWorkflowEnv(
            registry,
            runtime=_hotpot_semantic_runtime(registry, gateway),
            graph=graph,
            problem="What is the capital of France?",
            semantic_protocol="hotpotqa_verified_answer_slot_v1",
            recovery_policy="preserve_diagnose_repair_augment",
            required_evidence_tool_id="qa-retrieval",
        )
        env._failed_agent_ids.add("reasoner")
        env._diagnosed_unusable_agent_ids.add("reasoner")
        env._unresolved_dirty_agents.add("reasoner")
        env._progressive_outputs["replacement_reasoner"] = "Paris"
        env._progressive_output_metadata["replacement_reasoner"] = {}

        invalid_issue = env._delete_admission_issue("reasoner")

        self.assertIsNotNone(invalid_issue)
        artifact = {
            "question_scope": "What is the capital of France?",
            "answer_slot": {
                "answer_type": "short_answer",
                "answer_cardinality": "single",
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

    async def test_recovery_policy_protects_unreachable_artifact_without_failed_node(
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

        self.assertFalse(deleted.accepted)
        self.assertIn("replacement artifact takeover is required", deleted.feedback)
        self.assertTrue(env.graph.has_node("redundant"))
        self.assertTrue(env.graph.has_node("out"))

    async def test_hotpot_recovery_protects_disconnected_evidence_without_failure(
        self,
    ) -> None:
        graph = _hotpot_semantic_graph()
        graph.add_agent(
            AgentNode(
                "duplicate_reasoner",
                "balanced",
                "superseded semantic candidate",
                role_family="reasoner",
                allowed_tools=(QA_RETRIEVAL_TOOL_ID,),
                execution_mode="react",
                artifact_type="semantic_candidate",
            )
        )
        graph.add_agent(
            AgentNode(
                "duplicate_verifier",
                "balanced",
                "superseded verification",
                role_family="verifier",
                artifact_type="verified_semantic_answer",
            )
        )
        graph.add_agent(
            AgentNode(
                "duplicate_formatter",
                "fast",
                _HOTPOTQA_FORMAT_CONTRACT,
                role_family="format",
                artifact_type="answer_wrapper",
            )
        )
        graph.set_relation("duplicate_reasoner", "duplicate_verifier", True, False)
        graph.set_relation("duplicate_verifier", "duplicate_formatter", True, False)
        registry = make_registry()
        gateway = _ImmediateGateway()
        env = AgentWorkflowEnv(
            registry,
            runtime=_hotpot_semantic_runtime(registry, gateway),
            graph=graph,
            problem="What is the capital of France?",
            semantic_protocol="hotpotqa_verified_answer_slot_v1",
            recovery_policy="preserve_diagnose_repair_augment",
            required_evidence_tool_id="qa-retrieval",
        )
        reasoner_artifact = json.dumps(
            {
                "question_scope": "What is the capital of France?",
                "answer_slot": {
                    "answer_type": "short_answer",
                    "answer_cardinality": "single",
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
        )
        verifier_artifact = (
            "Candidate answer: Paris\n"
            "Evidence supported: true\n"
            "Entity attribute binding correct: true\n"
            "Alias binding correct: true\n"
            "Answer type cardinality correct: true\n"
            "Multi-hop complete: true\n"
            "Minimal answer surface: true\n"
            "Scope preserved: true\n"
            "Verification status: supported"
        )
        env._progressive_outputs.update(
            {
                "reader": "retrieved evidence",
                "reasoner": reasoner_artifact,
                "verifier": verifier_artifact,
                "formatter": "<answer>Paris</answer>",
                "duplicate_reasoner": "superseded candidate",
                "duplicate_verifier": "superseded verification",
                "duplicate_formatter": "<answer>Paris</answer>",
            }
        )
        env._progressive_output_metadata["reader"] = {
            "tool_receipts": [
                {
                    "tool_id": "qa-retrieval",
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

        state = env.recovery_state()
        self.assertEqual(
            ["reasoner", "verifier", "formatter"],
            state["active_semantic_lineage_agent_ids"],
        )
        self.assertIn(
            "duplicate_reasoner",
            state["redundant_after_replacement_takeover_agent_ids"],
        )
        self.assertNotIn("duplicate_reasoner", state["deletable_agent_ids"])
        actions = env.model_admissible_action_types()
        self.assertNotIn("set_output", actions)
        targets = env.model_admissible_action_targets()
        if "modify_agent" in targets:
            self.assertTrue(
                {"reasoner", "verifier", "formatter"}.isdisjoint(
                    targets["modify_agent"]["agent_ids"]
                )
            )
        if "set_relation" in targets:
            for candidate in targets["set_relation"]["candidates"]:
                if (
                    candidate["source_id"] == "reasoner"
                    and candidate["target_id"] == "verifier"
                ):
                    self.assertTrue(candidate["source_to_target"])
                if (
                    candidate["source_id"] == "verifier"
                    and candidate["target_id"] == "formatter"
                ):
                    self.assertTrue(candidate["source_to_target"])

        output_change = await env.step(
            '{"action":"set_output","agent_id":"duplicate_formatter"}'
        )
        self.assertFalse(output_change.accepted)
        self.assertIn("Output identity must be preserved", output_change.feedback)
        edge_removal = await env.step(
            '{"action":"set_relation","source_id":"reasoner",'
            '"target_id":"verifier","source_to_target":false,'
            '"target_to_source":false}'
        )
        self.assertFalse(edge_removal.accepted)
        self.assertIn("semantic-lineage relation", edge_removal.feedback)
        lineage_modify = await env.step(
            '{"action":"modify_agent","agent_id":"reasoner",'
            '"contract":"change the candidate"}'
        )
        self.assertFalse(lineage_modify.accepted)
        self.assertIn("verified semantic lineage", lineage_modify.feedback)

        deleted = await env.step(
            '{"action":"delete_agent","agent_id":"duplicate_reasoner"}'
        )

        self.assertFalse(deleted.accepted)
        self.assertIn("replacement artifact takeover is required", deleted.feedback)
        self.assertTrue(env.graph.has_node("duplicate_reasoner"))
        self.assertTrue(env.graph.has_node("reasoner"))

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
