from __future__ import annotations

import asyncio
from dataclasses import replace
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
    _QA_EVIDENCE_RETRIEVER_RECOVERY_COMPLETION,
    _QA_EVIDENCE_RETRIEVER_RECOVERY_CONTRACT,
    _QA_LOCATION_REASONER_RECOVERY_COMPLETION,
    _QA_LOCATION_REASONER_RECOVERY_CONTRACT,
    _ReadReceiptText,
    _evidence_span_matches_read,
)
from src.interactive.director import director_validate_live_action_target_domains
from src.interactive.model_registry import (
    ModelRegistry,
    ModelRegistryError,
    ModelSpec,
    ProviderSpec,
)
from src.interactive.qa_tool_adapter import (
    QA_RETRIEVAL_TOOL_ID,
    QA_VERIFIED_ANSWER_LINEAGE_PROTOCOL,
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
        "MARGARET 'Peggy' Seeger is an American folksinger. She married Ewan MacColl.",
        passage,
    )
    assert not _evidence_span_matches_read(
        "Peggy Seeger had American nationality and was Ewan MacColl's spouse.",
        passage,
    )


def test_artifact_version_binding_rejects_stale_semantic_input() -> None:
    current = {
        "reasoner": {"artifact_version": "reasoner:revision"},
        "verifier": {
            "artifact_version": "verifier:revision",
            "input_artifact_versions": {"reasoner": "reasoner:revision"},
        },
    }
    assert AgentWorkflowEnv._artifact_version_binding_issue(
        current,
        producer_id="reasoner",
        consumer_id="verifier",
        consumer_role="Verifier",
    ) is None

    stale = {
        **current,
        "verifier": {
            **current["verifier"],
            "input_artifact_versions": {"reasoner": "reasoner:draft"},
        },
    }
    issue = AgentWorkflowEnv._artifact_version_binding_issue(
        stale,
        producer_id="reasoner",
        consumer_id="verifier",
        consumer_role="Verifier",
    )
    assert issue is not None
    assert "not bound to the current artifact version" in issue


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


def _trivia_semantic_graph(*, reader_to_reasoner: bool = True) -> AgentGraph:
    """Return the shared QA lineage with a Tool-executed Retriever ingress."""

    relations = [
        AgentRelation("reasoner", "verifier", True, False),
        AgentRelation("verifier", "formatter", True, False),
    ]
    if reader_to_reasoner:
        relations.insert(
            0,
            AgentRelation("reader", "reasoner", True, False),
        )
    return AgentGraph(
        [
            AgentNode(
                "reader",
                "cheap",
                "retrieve answer-free evidence for the question entity and relation",
                role_family="evidence_retriever",
                allowed_tools=(QA_RETRIEVAL_TOOL_ID,),
                execution_mode="react",
                artifact_type="retrieval_evidence",
            ),
            AgentNode(
                "reasoner",
                "balanced",
                "bind grounded propositions to the requested answer slot",
                role_family="reasoner",
                allowed_tools=(QA_RETRIEVAL_TOOL_ID,),
                execution_mode="react",
                artifact_type="semantic_candidate",
            ),
            AgentNode(
                "verifier",
                "balanced",
                "verify evidence, entity, relation, scope, and answer lineage",
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
        relations,
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


def _trivia_semantic_runtime(
    registry: ModelRegistry,
    gateway: _ImmediateGateway,
) -> AgentRuntime:
    """Shared semantic-lineage fixture under the TriviaQA dataset binding."""

    return AgentRuntime(
        registry,
        gateway,
        execution_adapters={"react": ReasoningExecutionAdapter(gateway)},
        tool_registry=build_qa_tool_registry(_HotpotNoopRetrievalIndex()),
        dataset_id="triviaqa",
        semantic_protocol=QA_VERIFIED_ANSWER_LINEAGE_PROTOCOL,
    )


def _test_read_receipt(
    passage_id: str,
    *,
    text: str = "Paris is the capital of France.",
    title: str | None = None,
) -> dict[str, object]:
    passage: dict[str, object] = {
        "passage_id": passage_id,
        "text": text,
    }
    if title is not None:
        passage["title"] = title
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
                "passage_id": passage_id,
                "passage": passage,
            },
            "completed": True,
        },
        "error_type": None,
    }


def _test_evidence_retriever_artifact(passage_id: str) -> str:
    """Return a complete answer-free Retriever artifact for the shared fixture."""

    return json.dumps(
        {
            "question_scope": "What is the capital of France?",
            "entity_identity": {
                "question_surface": "France",
                "evidence_surface": "France",
            },
            "target_relation": "capital of",
            "answer_type_constraint": "short_answer",
            "evidence_proposition": {
                "subject": "Paris",
                "predicate": "is the capital of",
                "object_or_attribute_value": "France",
            },
            "evidence_span": "Paris is the capital of France.",
            "passage_id": passage_id,
        }
    )


def _test_retrieval_failure_record(
    graph: AgentGraph,
    *,
    agent_id: str,
    public_error_code: str = "retrieval_strategy_failure",
    strategies: tuple[str, ...] = (
        "initial_retrieval",
        "spelling_normalization",
    ),
    passage_ids: tuple[str, ...] = (),
    bounded_schedule_exhausted: bool = False,
) -> AgentFailureRecord:
    diagnosis = {
        "observation_status": "budget_exhausted",
        "public_error_code": public_error_code,
        "tool_plan_exhausted": True,
        "bounded_schedule_exhausted": bounded_schedule_exhausted,
        "retrieval_strategy_progress_count": len(strategies),
        "retrieval_strategy_schedule_prefix": list(strategies),
    }
    return AgentFailureRecord(
        request_id=f"{agent_id}-{public_error_code}",
        agent_id=agent_id,
        phase=ExecutionPhase.SINGLE,
        graph_revision=graph.revision,
        error_type="ReactExecutionError",
        message=f"react agent {agent_id!r} exhausted its bounded execution",
        metadata={
            "react_trace": [
                {
                    "turn": 1,
                    "observation_status": "budget_exhausted",
                    "terminal_failure_diagnosis": diagnosis,
                }
            ],
            "tool_receipts": [
                _test_read_receipt(passage_id)
                for passage_id in passage_ids
            ],
            "tool_plan_exhausted": True,
        },
    )


def _test_retrieval_turn_exhaustion_record(
    graph: AgentGraph,
    *,
    agent_id: str,
    strategies: tuple[str, ...] = (
        "initial_retrieval",
        "alias_expansion",
        "query_rewriting",
    ),
    passage_ids: tuple[str, ...] = ("public-read",),
    continuation_admissible: bool = True,
    tool_plan_exhausted: bool = False,
    bounded_schedule_exhausted: bool = False,
    remaining_tool_calls: int = 11,
    retrieval_attempts: tuple[dict[str, object], ...] = (),
) -> AgentFailureRecord:
    diagnosis = {
        "react_turn_exhausted": True,
        "continuation_admissible": continuation_admissible,
        "tool_plan_exhausted": tool_plan_exhausted,
        "bounded_schedule_exhausted": bounded_schedule_exhausted,
        "remaining_tool_calls": remaining_tool_calls,
        "retrieval_strategy_progress_count": len(strategies),
        "retrieval_strategy_schedule_prefix": list(strategies),
        "retrieval_attempts": list(retrieval_attempts),
    }
    return AgentFailureRecord(
        request_id=f"{agent_id}-react-turn-exhaustion",
        agent_id=agent_id,
        phase=ExecutionPhase.SINGLE,
        graph_revision=graph.revision,
        error_type="ReactExecutionError",
        message=f"react agent {agent_id!r} exhausted 32 turns",
        metadata={
            "react_trace": [
                {
                    "turn": 32,
                    "observation_status": "schema_invalid",
                    "public_error_code": (
                        "qa_retrieval_query_named_scope_loss: "
                        'missing_named_constraints=["american"]'
                    ),
                    "repair_instruction": (
                        "Preserve public Tool receipts and continue the "
                        "admitted retrieval schedule."
                    ),
                    "terminal_failure_diagnosis": diagnosis,
                }
            ],
            "tool_receipts": [
                _test_read_receipt(passage_id)
                for passage_id in passage_ids
            ],
            "tool_plan_exhausted": tool_plan_exhausted,
        },
    )


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
                                    "passage_id": "p1",
                                    "passage": {
                                        "passage_id": "p1",
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
                            "relation": "is a country in",
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

        availability = env.model_availability_receipt()
        self.assertFalse(availability["catalog_mutated"])
        self.assertEqual(["balanced"], availability["unavailable_model_ids"])
        self.assertNotIn("balanced", availability["available_model_ids"])
        self.assertEqual(
            ("reasoner", "verifier"),
            env._mandatory_repair_agent_ids(),
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
        self.assertEqual(
            ["balanced"],
            env.model_availability_receipt()["unavailable_model_ids"],
        )
        self.assertEqual(("verifier",), env._mandatory_repair_agent_ids())

    def test_http_400_does_not_quarantine_catalog_model(self) -> None:
        registry = make_multi_provider_registry()
        graph = _hotpot_semantic_graph()
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
                    request_id="request-400",
                    agent_id="reasoner",
                    phase=ExecutionPhase.SINGLE,
                    graph_revision=graph.revision,
                    error_type="OpenAICompatibleGatewayError",
                    message="provider request failed with HTTP status 400",
                ),
            ),
            current_agent_ids={node.id for node in graph.nodes},
        )

        self.assertEqual(
            [],
            env.model_availability_receipt()["unavailable_model_ids"],
        )

    def test_late_react_http_400_preserves_continuation_repair_boundary(
        self,
    ) -> None:
        registry = make_registry()
        graph = AgentGraph(
            [
                AgentNode(
                    "retriever",
                    "balanced",
                    "retrieve public evidence",
                    role_family="evidence_retriever",
                    allowed_tools=(QA_RETRIEVAL_TOOL_ID,),
                    execution_mode="react",
                    artifact_type="retrieval_evidence",
                )
            ]
        )
        env = AgentWorkflowEnv(
            registry,
            runtime=_trivia_semantic_runtime(registry, _ImmediateGateway()),
            graph=graph,
            problem="Who wrote the novel?",
            execute_on_edit=False,
            semantic_protocol=QA_VERIFIED_ANSWER_LINEAGE_PROTOCOL,
            recovery_policy="preserve_diagnose_repair_augment",
            required_evidence_tool_id=QA_RETRIEVAL_TOOL_ID,
        )
        record = AgentFailureRecord(
            request_id="request-late-400",
            agent_id="retriever",
            phase=ExecutionPhase.SINGLE,
            graph_revision=graph.revision,
            error_type="ReactGenerationError",
            message=(
                "OpenAICompatibleGatewayError: provider request failed "
                "with HTTP status 400"
            ),
            metadata={
                "http_status": 400,
                "react_trace": [
                    {
                        "turn": 1,
                        "observation": {
                            "observation_status": "success",
                            "result": {
                                "operation": "search",
                                "query": "novel author",
                                "top_k": 5,
                                "passage_ids": ["p1"],
                            },
                        },
                    }
                ],
                "tool_receipts": [
                    {
                        "tool_id": QA_RETRIEVAL_TOOL_ID,
                        "request": {
                            "action": "search",
                            "arguments": {"query": "novel author", "limit": 5},
                        },
                        "result": {"completed": True, "value": {}},
                        "error_type": None,
                    }
                ],
                "model_calls": [
                    {"request_status": "completed"},
                    {
                        "request_status": "failed",
                        "error_type": "OpenAICompatibleGatewayError",
                    },
                ],
            },
        )

        self.assertEqual(
            (
                "react_continuation_request_failure",
                "preserve_public_continuation",
                400,
            ),
            env._execution_failure_diagnosis(record),
        )
        env._record_failure_state(
            (record,),
            current_agent_ids={"retriever"},
        )
        self.assertFalse(env._provider_repair_required("retriever"))
        feedback = json.loads(
            env._execution_error_feedback(
                AgentRuntimeError(
                    "late bounded ReAct request failed",
                    failure_records=(record,),
                )
            ).split("=", 1)[1]
        )
        attributed = feedback["failed_agents"][0]
        self.assertEqual(
            "react_continuation_request_failure",
            attributed["failure_category"],
        )
        self.assertEqual(
            "contract",
            attributed["preferred_repair"]["field"],
        )
        self.assertNotEqual(
            "model_id",
            attributed["preferred_repair"]["field"],
        )

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
                    "passage_id": "p1",
                    "passage": {"passage_id": "p1", "text": passage},
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
                        "terminal_failure_diagnosis": {
                            "observation_status": "budget_exhausted",
                            "public_error_code": "retrieval_strategy_failure",
                            "tool_plan_exhausted": True,
                            "bounded_schedule_exhausted": False,
                            "retrieval_attempt_count": 2,
                            "retrieval_strategy_progress_count": 1,
                            "recall_expansion_count": 1,
                            "retrieval_strategy_schedule_prefix": [
                                "initial_retrieval"
                            ],
                            "normalized_query_novelty_verified": True,
                            "strategy_semantics_verified": False,
                            "successful_search_with_hits_count": 2,
                            "successful_empty_search_count": 0,
                            "tool_error_count": 0,
                            "search_queries": ["private from compact summary"],
                        },
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
        self.assertEqual(
            "retrieval_strategy_failure",
            attributed["failure_category"],
        )
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
            1,
            summary["public_error_code_counts"]["retrieval_strategy_failure"],
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
        self.assertEqual(
            {
                "observation_status": "budget_exhausted",
                "public_error_code": "retrieval_strategy_failure",
                "tool_plan_exhausted": True,
                "bounded_schedule_exhausted": False,
                "retrieval_attempt_count": 2,
                "retrieval_strategy_progress_count": 1,
                "recall_expansion_count": 1,
                "normalized_query_novelty_verified": True,
                "strategy_semantics_verified": False,
                "successful_search_with_hits_count": 2,
                "successful_empty_search_count": 0,
                "tool_error_count": 0,
                "retrieval_strategy_schedule_prefix": ["initial_retrieval"],
            },
            summary["terminal_failure_diagnosis"],
        )
        self.assertNotIn("search_queries", summary["terminal_failure_diagnosis"])
        self.assertNotIn(passage, feedback_text)

    def test_react_summary_preserves_turn_exhaustion_continuation(self) -> None:
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
        diagnosis = {
            "react_turn_exhausted": True,
            "tool_plan_exhausted": False,
            "bounded_schedule_exhausted": False,
            "continuation_admissible": True,
            "remaining_tool_calls": 9,
            "retrieval_strategy_progress_count": 2,
            "verified_retrieval_strategy_coverage": [
                "initial_retrieval",
                "spelling_normalization",
            ],
            "missing_retrieval_strategy_coverage": [
                "alias_expansion",
                "entity_disambiguation",
                "query_rewriting",
            ],
        }
        record = AgentFailureRecord(
            request_id="request-turn-exhaustion",
            agent_id="retriever",
            phase=ExecutionPhase.SINGLE,
            graph_revision=graph.revision,
            error_type="ReactExecutionError",
            message="react agent exhausted its bounded turn window",
            metadata={
                "react_trace": [
                    {"terminal_failure_diagnosis": diagnosis}
                ]
            },
        )

        summary = env._react_public_error_summary(record)
        self.assertEqual(
            diagnosis,
            summary["terminal_failure_diagnosis"],
        )

    def test_failure_diagnosis_promotes_only_typed_terminal_receipts(
        self,
    ) -> None:
        expected = {
            "knowledge_base_coverage_failure": (
                "repair_retrieval_or_database_coverage"
            ),
            "retrieval_recall_failure": (
                "repair_retrieval_or_database_coverage"
            ),
            "retrieval_strategy_failure": (
                "repair_execution_contract_or_tool_plan"
            ),
        }
        for index, (category, retryability) in enumerate(expected.items()):
            diagnosis = {
                "observation_status": "budget_exhausted",
                "public_error_code": category,
            }
            trace_entry = (
                {"terminal_failure_diagnosis": diagnosis}
                if index != 1
                else {
                    "observation": {
                        "terminal_failure_diagnosis": diagnosis,
                    }
                }
            )
            record = AgentFailureRecord(
                request_id=f"typed-retrieval-{index}",
                agent_id="reader",
                phase=ExecutionPhase.SINGLE,
                graph_revision=0,
                error_type="ReactExecutionError",
                message="bounded ReAct execution failed",
                metadata={"react_trace": [trace_entry]},
            )

            with self.subTest(category=category):
                self.assertEqual(
                    (category, retryability, None),
                    AgentWorkflowEnv._execution_failure_diagnosis(record),
                )

        untyped = AgentFailureRecord(
            request_id="untyped-retrieval-text",
            agent_id="reader",
            phase=ExecutionPhase.SINGLE,
            graph_revision=0,
            error_type="RuntimeError",
            message="knowledge_base_coverage_failure",
        )
        self.assertEqual(
            "execution_contract_or_runtime_failure",
            AgentWorkflowEnv._execution_failure_diagnosis(untyped)[0],
        )

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
            ["evidence_retriever"],
            add_domain["admitted_new_role_families"],
        )

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

    def test_reasoner_continuation_is_not_handed_to_cross_role_retriever(
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

        self.assertEqual({}, env._recovery_continuation_handoff(action))
        self.assertNotIn("repair_reader", env._failure_continuations)
        self.assertNotIn(
            "continuation_source_agent_id",
            env._failure_continuations["reasoner"],
        )
        self.assertEqual(
            receipts,
            env._failure_continuations["reasoner"]["tool_receipts"],
        )
        self.assertEqual(
            trace,
            env._failure_continuations["reasoner"]["react_trace"],
        )

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
        self.assertEqual(
            {
                "contract": (
                    "align facts to answer slot and select semantic answer"
                ),
                "completion_condition": None,
            },
            modify_domain["per_agent_candidates"][0]["current_values"],
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
        all_relation_candidates = env._all_model_admissible_relation_candidates()
        self.assertGreater(len(all_relation_candidates), 1)
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
        # This is a recovery boundary, but the Reasoner itself is not
        # repair-exhausted. The missing responsibility must therefore be
        # selected by the general semantic Canvas mask, not by the old
        # exhausted-Reasoner-only branch.
        env._failed_agent_ids.add("reader")
        env._repair_exhausted_agent_ids.add("reader")

        self.assertEqual((), env._repair_exhausted_reasoner_ids())
        self.assertTrue(env._all_model_admissible_relation_candidates())
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

    def test_trivia_failure_feedback_does_not_advertise_roles_outside_live_add_domain(
        self,
    ) -> None:
        complete = _trivia_semantic_graph()
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
            runtime=_trivia_semantic_runtime(registry, _ImmediateGateway()),
            graph=graph,
            problem="Who wrote the novel?",
            execute_on_edit=False,
            max_agents=8,
            semantic_protocol=QA_VERIFIED_ANSWER_LINEAGE_PROTOCOL,
            recovery_policy="preserve_diagnose_repair_augment",
            required_evidence_tool_id=QA_RETRIEVAL_TOOL_ID,
        )
        record = _test_retrieval_failure_record(
            graph,
            agent_id="reader",
            passage_ids=("preserved-read",),
        )
        env._failed_agent_ids.add("reader")
        env._repair_exhausted_agent_ids.add("reader")
        env._latest_failure_record_by_agent["reader"] = record

        self.assertEqual(("add_subgraph",), env.model_admissible_action_types())
        self.assertEqual(
            ["evidence_retriever"],
            env.model_admissible_action_targets()["add_subgraph"][
                "admitted_new_role_families"
            ],
        )
        feedback = json.loads(
            env._execution_error_feedback(
                AgentRuntimeError(
                    "bounded Retriever failed",
                    failure_records=(record,),
                )
            ).split("=", 1)[1]
        )
        self.assertEqual(
            ["add_subgraph"],
            feedback["failed_agents"][0]["preferred_repair"][
                "action_order"
            ],
        )
        self.assertEqual(
            ["evidence_retriever"],
            feedback["failed_agents"][0]["preferred_repair"][
                "admitted_role_families"
            ],
        )
        self.assertEqual(
            ["add_subgraph"],
            feedback["recovery_state"]["preferred_actions"],
        )

    async def test_full_capacity_dead_auxiliary_delete_admits_reasoner_recovery_unit(
        self,
    ) -> None:
        complete = _trivia_semantic_graph()
        graph = AgentGraph(
            [node for node in complete.nodes if node.id != "formatter"],
            [
                relation
                for relation in complete.relations
                if "formatter" not in {relation.source_id, relation.target_id}
            ],
        )
        for agent_id in (
            "dead_reader",
            "auxiliary_1",
            "auxiliary_2",
            "auxiliary_3",
            "auxiliary_4",
        ):
            graph.add_agent(
                AgentNode(
                    agent_id,
                    "cheap",
                    "retrieve bounded supplementary evidence",
                    role_family="evidence_retriever",
                    allowed_tools=(QA_RETRIEVAL_TOOL_ID,),
                    execution_mode="react",
                    artifact_type="retrieval_evidence",
                )
            )
        registry = make_registry()
        env = AgentWorkflowEnv(
            registry,
            runtime=_trivia_semantic_runtime(registry, _ImmediateGateway()),
            graph=graph,
            problem="Who wrote the novel?",
            execute_on_edit=False,
            max_agents=8,
            semantic_protocol=QA_VERIFIED_ANSWER_LINEAGE_PROTOCOL,
            recovery_policy="preserve_diagnose_repair_augment",
            required_evidence_tool_id=QA_RETRIEVAL_TOOL_ID,
        )
        env._failed_agent_ids.update({"dead_reader", "reasoner"})
        env._repair_exhausted_agent_ids.update(
            {"dead_reader", "reasoner"}
        )
        env._latest_failure_record_by_agent["dead_reader"] = AgentFailureRecord(
            request_id="dead-reader-bounded-exhaustion",
            agent_id="dead_reader",
            phase=ExecutionPhase.SINGLE,
            graph_revision=graph.revision,
            error_type="ReactExecutionError",
            message="react agent 'dead_reader' exhausted 8 turns",
        )

        self.assertEqual(8, len(env.graph.nodes))
        self.assertEqual(("delete_agent",), env.model_admissible_action_types())
        self.assertEqual(
            ["dead_reader"],
            env.model_admissible_action_targets()["delete_agent"]["agent_ids"],
        )

        deleted = await env.step(
            '{"action":"delete_agent","agent_id":"dead_reader"}'
        )

        self.assertTrue(deleted.accepted, deleted.feedback)
        self.assertEqual(7, len(env.graph.nodes))
        self.assertEqual(1, len(env.history))
        self.assertEqual("delete_agent", env.history[0].action.action_type.value)
        self.assertEqual(("add_subgraph",), env.model_admissible_action_types())
        add_domain = env.model_admissible_action_targets()["add_subgraph"]
        self.assertEqual(
            ["evidence_retriever"],
            add_domain["admitted_new_role_families"],
        )

        added = await env.step(
            json.dumps(
                {
                    "action": "add_subgraph",
                    "agents": [
                        {
                            "agent_id": "replacement_reader",
                            "model_id": "cheap",
                            "contract": (
                                _QA_EVIDENCE_RETRIEVER_RECOVERY_CONTRACT
                            ),
                            "completion_condition": (
                                _QA_EVIDENCE_RETRIEVER_RECOVERY_COMPLETION
                            ),
                            "role_family": "evidence_retriever",
                            "allowed_tools": [QA_RETRIEVAL_TOOL_ID],
                            "execution_mode": "react",
                            "artifact_type": "retrieval_evidence",
                        }
                    ],
                    "relations": [],
                    "output_agent_id": None,
                }
            )
        )

        self.assertTrue(added.accepted, added.feedback)
        self.assertEqual(8, len(env.graph.nodes))
        self.assertEqual(2, len(env.history))
        self.assertEqual("delete_agent", env.history[0].action.action_type.value)
        self.assertEqual("add_subgraph", env.history[1].action.action_type.value)
        self.assertEqual(("modify_agent",), env.model_admissible_action_types())

    def test_capacity_delete_preservation_predicate_counterexamples(self) -> None:
        def make_env() -> AgentWorkflowEnv:
            complete = _trivia_semantic_graph()
            graph = AgentGraph(
                [node for node in complete.nodes if node.id != "formatter"],
                [
                    relation
                    for relation in complete.relations
                    if "formatter"
                    not in {relation.source_id, relation.target_id}
                ],
            )
            for agent_id in (
                "dead_reader",
                "auxiliary_1",
                "auxiliary_2",
                "auxiliary_3",
                "auxiliary_4",
            ):
                graph.add_agent(
                    AgentNode(
                        agent_id,
                        "cheap",
                        "retrieve bounded supplementary evidence",
                        role_family="evidence_retriever",
                        allowed_tools=(QA_RETRIEVAL_TOOL_ID,),
                        execution_mode="react",
                        artifact_type="retrieval_evidence",
                    )
                )
            registry = make_registry()
            env = AgentWorkflowEnv(
                registry,
                runtime=_trivia_semantic_runtime(
                    registry,
                    _ImmediateGateway(),
                ),
                graph=graph,
                problem="Who wrote the novel?",
                execute_on_edit=False,
                max_agents=8,
                semantic_protocol=QA_VERIFIED_ANSWER_LINEAGE_PROTOCOL,
                recovery_policy="preserve_diagnose_repair_augment",
                required_evidence_tool_id=QA_RETRIEVAL_TOOL_ID,
            )
            env._failed_agent_ids.update({"dead_reader", "reasoner"})
            env._repair_exhausted_agent_ids.update(
                {"dead_reader", "reasoner"}
            )
            env._latest_failure_record_by_agent[
                "dead_reader"
            ] = AgentFailureRecord(
                request_id="dead-reader-bounded-exhaustion",
                agent_id="dead_reader",
                phase=ExecutionPhase.SINGLE,
                graph_revision=graph.revision,
                error_type="ReactExecutionError",
                message="react agent 'dead_reader' exhausted 8 turns",
            )
            return env

        baseline = make_env()
        self.assertEqual(
            ("dead_reader",),
            baseline._capacity_blocking_failed_auxiliary_delete_ids(),
        )

        current_artifact = make_env()
        current_artifact._progressive_outputs["dead_reader"] = "retained artifact"
        self.assertEqual(
            (), current_artifact._capacity_blocking_failed_auxiliary_delete_ids()
        )

        previous_artifact = make_env()
        previous_artifact._previous_revision_outputs[
            "dead_reader"
        ] = "retained prior artifact"
        self.assertEqual(
            (), previous_artifact._capacity_blocking_failed_auxiliary_delete_ids()
        )

        retained_read = make_env()
        retained_read._failure_continuations["dead_reader"] = {
            "tool_receipts": [_test_read_receipt("retained-public-read")]
        }
        self.assertEqual(
            (), retained_read._capacity_blocking_failed_auxiliary_delete_ids()
        )
        self.assertEqual([], retained_read._model_admissible_relation_candidates())
        self.assertEqual((), retained_read.model_admissible_action_types())

        incoming_edge = make_env()
        incoming_edge.graph.set_relation(
            "auxiliary_1", "dead_reader", True, False
        )
        self.assertEqual(
            (), incoming_edge._capacity_blocking_failed_auxiliary_delete_ids()
        )

        outgoing_edge = make_env()
        outgoing_edge.graph.set_relation(
            "dead_reader", "auxiliary_1", True, False
        )
        self.assertEqual(
            (), outgoing_edge._capacity_blocking_failed_auxiliary_delete_ids()
        )

        repair_not_exhausted = make_env()
        repair_not_exhausted._repair_exhausted_agent_ids.clear()
        self.assertEqual(
            (), repair_not_exhausted._capacity_blocking_failed_auxiliary_delete_ids()
        )

        unbounded_failure = make_env()
        unbounded_failure._latest_failure_record_by_agent[
            "dead_reader"
        ] = AgentFailureRecord(
            request_id="provider-failure",
            agent_id="dead_reader",
            phase=ExecutionPhase.SINGLE,
            graph_revision=unbounded_failure.graph.revision,
            error_type="RuntimeError",
            message="provider request failed",
        )
        self.assertEqual(
            (), unbounded_failure._capacity_blocking_failed_auxiliary_delete_ids()
        )

        spare_capacity = make_env()
        spare_capacity.max_agents = 9
        self.assertEqual(
            (), spare_capacity._capacity_blocking_failed_auxiliary_delete_ids()
        )

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
        env._progressive_outputs["repair_evidence"] = (
            _test_evidence_retriever_artifact("p1")
        )
        env._progressive_output_metadata["repair_evidence"] = {
            "tool_receipts": [_test_read_receipt("p1")]
        }

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
                outputs={
                    "repair_evidence": _test_evidence_retriever_artifact("p1")
                },
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

    def test_structured_reasoner_failure_with_valid_ingress_fails_closed(
        self,
    ) -> None:
        complete = _trivia_semantic_graph()
        graphs = (
            AgentGraph(
                [
                    node
                    for node in complete.nodes
                    if node.id in {"reader", "reasoner"}
                ],
                [AgentRelation("reader", "reasoner", True, False)],
            ),
            complete,
        )
        for graph in graphs:
            with self.subTest(complete_spine=graph.output_agent_id is not None):
                registry = make_registry()
                env = AgentWorkflowEnv(
                    registry,
                    runtime=_trivia_semantic_runtime(
                        registry,
                        _ImmediateGateway(),
                    ),
                    graph=graph,
                    problem="What is the capital of France?",
                    execute_on_edit=False,
                    max_agents=8,
                    semantic_protocol=QA_VERIFIED_ANSWER_LINEAGE_PROTOCOL,
                    recovery_policy="preserve_diagnose_repair_augment",
                    required_evidence_tool_id=QA_RETRIEVAL_TOOL_ID,
                )
                env._progressive_outputs["reader"] = (
                    _test_evidence_retriever_artifact("reader-public")
                )
                env._progressive_output_metadata["reader"] = {
                    "artifact_version": "reader:v1",
                    "tool_receipts": [_test_read_receipt("reader-public")],
                }
                failure = AgentFailureRecord(
                    request_id="reasoner-answer-slot-failure",
                    agent_id="reasoner",
                    phase=ExecutionPhase.SINGLE,
                    graph_revision=graph.revision,
                    error_type="ReactExecutionError",
                    message="react agent 'reasoner' exhausted 20 turns",
                    metadata={
                        "react_trace": [
                            {
                                "turn": 10,
                                "observation_status": "budget_exhausted",
                                "terminal_failure_diagnosis": {
                                    "public_error_code": (
                                        "retrieval_recall_failure"
                                    ),
                                    "bounded_schedule_exhausted": True,
                                },
                            },
                            {
                                "turn": 20,
                                "observation_status": "schema_invalid",
                                "public_error_code": (
                                    "qa_semantic_artifact_invalid: Reasoner "
                                    "answer_slot.answer_field selects 'subject'"
                                ),
                            }
                        ],
                        "tool_receipts": [_test_read_receipt("reasoner-public")],
                        "input_artifact_versions": {"reader": "reader:v1"},
                    },
                )
                env._failed_agent_ids.add("reasoner")
                env._react_exhausted_agent_ids.add("reasoner")
                env._repair_exhausted_agent_ids.add("reasoner")
                env._latest_failure_record_by_agent["reasoner"] = failure

                self.assertEqual(
                    ("reader",),
                    env._receipt_valid_routed_evidence_retriever_ids("reasoner"),
                )
                self.assertFalse(
                    env._reasoner_failure_requires_evidence_augmentation(
                        "reasoner"
                    )
                )
                self.assertEqual((), env.model_admissible_action_types())
                self.assertEqual({}, env.model_admissible_action_targets())

    def test_typed_reasoner_retrieval_deficit_admits_one_bounded_retriever(
        self,
    ) -> None:
        graph = _trivia_semantic_graph()
        registry = make_registry()
        env = AgentWorkflowEnv(
            registry,
            runtime=_trivia_semantic_runtime(registry, _ImmediateGateway()),
            graph=graph,
            problem="What is the capital of France?",
            execute_on_edit=False,
            max_agents=8,
            semantic_protocol=QA_VERIFIED_ANSWER_LINEAGE_PROTOCOL,
            recovery_policy="preserve_diagnose_repair_augment",
            required_evidence_tool_id=QA_RETRIEVAL_TOOL_ID,
        )
        for agent_id in ("reader",):
            env._progressive_outputs[agent_id] = (
                _test_evidence_retriever_artifact(f"{agent_id}-public")
            )
            env._progressive_output_metadata[agent_id] = {
                "tool_receipts": [_test_read_receipt(f"{agent_id}-public")]
            }
        failure = _test_retrieval_failure_record(
            graph,
            agent_id="reasoner",
            public_error_code="retrieval_recall_failure",
            bounded_schedule_exhausted=False,
        )
        env._failed_agent_ids.add("reasoner")
        env._react_exhausted_agent_ids.add("reasoner")
        env._repair_exhausted_agent_ids.add("reasoner")
        env._latest_failure_record_by_agent["reasoner"] = failure

        self.assertEqual(("add_subgraph",), env.model_admissible_action_types())
        self.assertEqual(
            ["evidence_retriever"],
            env.model_admissible_action_targets()["add_subgraph"][
                "admitted_new_role_families"
            ],
        )

        env.graph.add_agent(
            AgentNode(
                "repair_reader",
                "cheap",
                "retrieve another receipt-grounded evidence proposition",
                role_family="evidence_retriever",
                allowed_tools=(QA_RETRIEVAL_TOOL_ID,),
                execution_mode="react",
                artifact_type="retrieval_evidence",
            )
        )
        env.graph.set_relation("repair_reader", "reasoner", True, False)
        env._progressive_outputs["repair_reader"] = (
            _test_evidence_retriever_artifact("repair-public")
        )
        env._progressive_output_metadata["repair_reader"] = {
            "tool_receipts": [_test_read_receipt("repair-public")]
        }

        self.assertFalse(
            env._reasoner_failure_requires_evidence_augmentation("reasoner")
        )
        self.assertEqual((), env.model_admissible_action_types())

    def test_reciprocal_receipt_valid_ingress_closes_structured_recovery(
        self,
    ) -> None:
        graph = _trivia_semantic_graph()
        graph.set_relation("reader", "reasoner", True, True)
        registry = make_registry()
        env = AgentWorkflowEnv(
            registry,
            runtime=_trivia_semantic_runtime(registry, _ImmediateGateway()),
            graph=graph,
            problem="What is the capital of France?",
            execute_on_edit=False,
            max_agents=8,
            semantic_protocol=QA_VERIFIED_ANSWER_LINEAGE_PROTOCOL,
            recovery_policy="preserve_diagnose_repair_augment",
            required_evidence_tool_id=QA_RETRIEVAL_TOOL_ID,
        )
        env._progressive_outputs["reader"] = (
            _test_evidence_retriever_artifact("reader-public")
        )
        env._progressive_output_metadata["reader"] = {
            "artifact_version": "reader:v1",
            "tool_receipts": [_test_read_receipt("reader-public")],
        }
        failure = AgentFailureRecord(
            request_id="reasoner-reciprocal-answer-slot-failure",
            agent_id="reasoner",
            phase=ExecutionPhase.SINGLE,
            graph_revision=graph.revision,
            error_type="ReactExecutionError",
            message="reasoner structured artifact rejected",
            metadata={
                "react_trace": [
                    {
                        "turn": 20,
                        "observation_status": "schema_invalid",
                        "public_error_code": (
                            "qa_semantic_artifact_invalid: "
                            "qa_location_containment_lineage_missing"
                        ),
                    }
                ],
                "tool_receipts": [_test_read_receipt("reasoner-public")],
                "input_artifact_versions": {"reader": "reader:v1"},
            },
        )
        env._failed_agent_ids.add("reasoner")
        env._react_exhausted_agent_ids.add("reasoner")
        env._repair_exhausted_agent_ids.add("reasoner")
        env._latest_failure_record_by_agent["reasoner"] = failure

        self.assertEqual(
            ("reader",),
            env._receipt_valid_routed_evidence_retriever_ids("reasoner"),
        )
        self.assertFalse(
            env._reasoner_failure_requires_evidence_augmentation("reasoner")
        )
        self.assertEqual((), env.model_admissible_action_types())

    async def test_triviaqa_location_reasoner_repair_is_receipt_conditioned_and_bounded(
        self,
    ) -> None:
        question = "Where in England was Dame Judi Dench born?"
        birth_span = (
            "Dench was born in Heworth, North Riding of Yorkshire."
        )
        containment_span = (
            "Heworth, York Heworth is part of the city of York in North "
            "Yorkshire, England."
        )

        graph = _trivia_semantic_graph()
        registry = make_registry()
        env = AgentWorkflowEnv(
            registry,
            runtime=_trivia_semantic_runtime(registry, _ImmediateGateway()),
            graph=graph,
            problem=question,
            execute_on_edit=False,
            semantic_protocol=QA_VERIFIED_ANSWER_LINEAGE_PROTOCOL,
            recovery_policy="preserve_diagnose_repair_augment",
            required_evidence_tool_id=QA_RETRIEVAL_TOOL_ID,
            require_format_agent=True,
        )
        birth_receipt = _test_read_receipt(
            "dench-birthplace",
            title="Judi Dench",
            text=birth_span,
        )
        env._progressive_outputs["reader"] = json.dumps(
            {
                "question_scope": question,
                "entity_identity": {
                    "question_surface": "Dame Judi Dench",
                    "evidence_surface": "Dench",
                },
                "target_relation": "be born in",
                "answer_type_constraint": "location",
                "evidence_proposition": {
                    "subject": "Dench",
                    "predicate": "born in",
                    "object_or_attribute_value": (
                        "Heworth, North Riding of Yorkshire"
                    ),
                },
                "evidence_span": birth_span,
                "passage_id": "dench-birthplace",
            }
        )
        env._progressive_output_metadata["reader"] = {
            "artifact_version": "reader:v1",
            "tool_receipts": [birth_receipt],
        }
        failure = AgentFailureRecord(
            request_id="reasoner-location-containment-repair",
            agent_id="reasoner",
            phase=ExecutionPhase.SINGLE,
            graph_revision=graph.revision,
            error_type="ReactExecutionError",
            message="react agent 'reasoner' exhausted 20 turns",
            metadata={
                "react_trace": [
                    {
                        "turn": 20,
                        "observation_status": "schema_invalid",
                        "public_error_code": (
                            "qa_semantic_artifact_invalid: "
                            "qa_location_containment_lineage_missing"
                        ),
                        "structured_action": {
                            "kind": "complete",
                            "name": "complete",
                            "arguments": {
                                "value": {
                                    "evidence_propositions": [
                                        {
                                            "subject": "Dench",
                                            "relation": "born in",
                                            "object_or_attribute_value": (
                                                "Heworth, North Riding of Yorkshire"
                                            ),
                                            "evidence_span": birth_span,
                                        }
                                    ]
                                }
                            },
                        },
                    }
                ],
                "tool_receipts": [
                    _test_read_receipt(
                        "heworth-york",
                        title="Heworth, York",
                        text=containment_span,
                    )
                ],
                "input_artifact_versions": {"reader": "reader:v1"},
                "tool_plan_exhausted": False,
            },
        )
        env._record_failure_state(
            (failure,),
            current_agent_ids={node.id for node in graph.nodes},
        )

        actions = env.model_admissible_action_types()
        targets = env.model_admissible_action_targets()
        self.assertEqual(("modify_agent",), actions)
        director_validate_live_action_target_domains(actions, targets)
        candidate = targets["modify_agent"]["per_agent_candidates"][0]
        self.assertEqual("reasoner", candidate["agent_id"])
        self.assertEqual(
            [_QA_LOCATION_REASONER_RECOVERY_CONTRACT],
            candidate["discrete_value_domains"]["contract"],
        )
        self.assertEqual(
            [_QA_LOCATION_REASONER_RECOVERY_COMPLETION],
            candidate["discrete_value_domains"]["completion_condition"],
        )

        unrelated_env = AgentWorkflowEnv(
            registry,
            runtime=_trivia_semantic_runtime(registry, _ImmediateGateway()),
            graph=graph,
            problem=question,
            execute_on_edit=False,
            semantic_protocol=QA_VERIFIED_ANSWER_LINEAGE_PROTOCOL,
            recovery_policy="preserve_diagnose_repair_augment",
            required_evidence_tool_id=QA_RETRIEVAL_TOOL_ID,
            require_format_agent=True,
        )
        unrelated_failure = replace(
            failure,
            request_id="reasoner-unrelated-containment-read",
            metadata={
                **failure.metadata,
                "tool_receipts": [
                    _test_read_receipt(
                        "greenwich-london",
                        title="Greenwich",
                        text=(
                            "Greenwich is part of the city of London in England."
                        ),
                    )
                ],
            },
        )
        unrelated_env._record_failure_state(
            (unrelated_failure,),
            current_agent_ids={node.id for node in graph.nodes},
        )
        self.assertEqual(
            {},
            unrelated_env._triviaqa_location_reasoner_recovery_field_values(
                "reasoner"
            ),
        )
        unrelated_targets = unrelated_env.model_admissible_action_targets()
        self.assertNotIn(
            "reasoner",
            {
                item["agent_id"]
                for item in unrelated_targets.get("modify_agent", {}).get(
                    "per_agent_candidates",
                    (),
                )
            },
        )

        for field_name, conflicting_value in (
            (
                "contract",
                "bind the first-hop location proposition's subject",
            ),
            (
                "completion_condition",
                "complete without further geographic confirmation",
            ),
        ):
            revision = env.revision
            rejected = await env.step(
                json.dumps(
                    {
                        "action": "modify_agent",
                        "agent_id": "reasoner",
                        field_name: conflicting_value,
                    }
                )
            )
            self.assertFalse(rejected.accepted)
            self.assertIn("receipt-conditioned", rejected.feedback)
            self.assertEqual(revision, env.revision)

        repaired = await env.step(
            json.dumps(
                {
                    "action": "modify_agent",
                    "agent_id": "reasoner",
                    "completion_condition": (
                        _QA_LOCATION_REASONER_RECOVERY_COMPLETION
                    ),
                }
            )
        )
        self.assertTrue(repaired.accepted, repaired.feedback)

        repeated_failure = replace(
            failure,
            request_id="reasoner-location-containment-repair-repeated",
            graph_revision=env.revision,
        )
        env._record_failure_state(
            (repeated_failure,),
            current_agent_ids={node.id for node in env.graph.nodes},
        )

        self.assertNotIn("reasoner", env._repair_exhausted_agent_ids)
        self.assertEqual(("modify_agent",), env.model_admissible_action_types())
        remaining_targets = env.model_admissible_action_targets()
        remaining_candidate = remaining_targets["modify_agent"][
            "per_agent_candidates"
        ][0]
        self.assertEqual(["contract"], remaining_candidate["mutable_fields"])
        self.assertEqual(
            [_QA_LOCATION_REASONER_RECOVERY_CONTRACT],
            remaining_candidate["discrete_value_domains"]["contract"],
        )

        second_repair = await env.step(
            json.dumps(
                {
                    "action": "modify_agent",
                    "agent_id": "reasoner",
                    "contract": _QA_LOCATION_REASONER_RECOVERY_CONTRACT,
                }
            )
        )
        self.assertTrue(second_repair.accepted, second_repair.feedback)
        exhausted_failure = replace(
            failure,
            request_id="reasoner-location-repair-after-both-fields",
            graph_revision=env.revision,
        )
        env._record_failure_state(
            (exhausted_failure,),
            current_agent_ids={node.id for node in env.graph.nodes},
        )

        self.assertIn("reasoner", env._repair_exhausted_agent_ids)
        self.assertFalse(
            env._reasoner_failure_requires_evidence_augmentation("reasoner")
        )
        self.assertEqual((), env.model_admissible_action_types())
        self.assertNotIn("add_subgraph", env.model_admissible_action_targets())

    async def test_exhausted_reasoner_adds_isolated_retriever_before_routing(
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
        env._latest_failure_record_by_agent["reasoner"] = (
            _react_exhaustion_record(
                graph,
                request_id="reasoner-first-retriever-augmentation",
            )
        )

        add_domain = env.model_admissible_action_targets()["add_subgraph"]
        self.assertEqual([], add_domain["relations"])
        self.assertIsNone(add_domain["output_agent_id"])
        self.assertEqual(
            ["evidence_retriever"],
            add_domain["admitted_new_role_families"],
        )

        add_payload = {
            "action": "add_subgraph",
            "agents": [
                {
                    "agent_id": "repair_retriever",
                    "model_id": "cheap",
                    "contract": "retrieve evidence for the requested relation",
                    "role_family": "evidence_retriever",
                    "allowed_tools": [QA_RETRIEVAL_TOOL_ID],
                    "execution_mode": "react",
                    "artifact_type": "retrieval_evidence",
                }
            ],
            "relations": [],
        }
        inline_relation = json.loads(json.dumps(add_payload))
        inline_relation["relations"] = [
            {
                "source_id": "repair_retriever",
                "target_id": "reasoner",
                "source_to_target": True,
                "target_to_source": False,
            }
        ]
        rejected = await env.step(json.dumps(inline_relation))
        self.assertFalse(rejected.accepted)
        self.assertIn("isolated Canvas unit", rejected.feedback)

        observed_execution: list[tuple[tuple[str, ...], str | None]] = []

        async def isolated_success(
            candidate_graph: AgentGraph,
            *args: object,
            **kwargs: object,
        ) -> AgentRuntimeResult:
            del args
            observed_execution.append(
                (
                    tuple(node.id for node in candidate_graph.nodes),
                    candidate_graph.output_agent_id,
                )
            )
            self.assertEqual(
                {"repair_retriever"},
                kwargs["dirty_agents"],
            )
            return AgentRuntimeResult(
                run_id="isolated-retriever-success",
                graph_revision=candidate_graph.revision,
                output_agent_id=None,
                final_answer=None,
                outputs={
                    "repair_retriever": _test_evidence_retriever_artifact(
                        "repair-public"
                    )
                },
                output_metadata={
                    "repair_retriever": {
                        "tool_receipts": [
                            _test_read_receipt("repair-public")
                        ]
                    }
                },
                calls=(),
                block_completion_order=(("repair_retriever",),),
                executed_agent_ids=("repair_retriever",),
            )

        with patch.object(runtime, "execute", side_effect=isolated_success):
            added = await env.step(json.dumps(add_payload))

        self.assertTrue(added.accepted, added.feedback)
        self.assertEqual([(("repair_retriever",), None)], observed_execution)
        self.assertEqual(("set_relation",), env.model_admissible_action_types())
        self.assertEqual(
            [
                {
                    "source_id": "repair_retriever",
                    "target_id": "reasoner",
                    "source_to_target": True,
                    "target_to_source": False,
                }
            ],
            env.model_admissible_action_targets()["set_relation"][
                "candidates"
            ],
        )

    def test_exhausted_reasoner_routes_new_evidence_before_output_selection(
        self,
    ) -> None:
        """A partial Canvas must not strand a successful recovery artifact."""

        complete = _trivia_semantic_graph()
        graph = AgentGraph(
            [node for node in complete.nodes if node.id != "formatter"],
            [
                relation
                for relation in complete.relations
                if "formatter"
                not in {relation.source_id, relation.target_id}
            ],
        )
        graph.add_agent(
            AgentNode(
                "repair_retriever",
                "cheap",
                "retrieve additional evidence for the requested relation",
                role_family="evidence_retriever",
                allowed_tools=(QA_RETRIEVAL_TOOL_ID,),
                execution_mode="react",
                artifact_type="retrieval_evidence",
            )
        )
        registry = make_registry()
        env = AgentWorkflowEnv(
            registry,
            runtime=_trivia_semantic_runtime(
                registry,
                _ImmediateGateway(),
            ),
            graph=graph,
            problem="What is the capital of France?",
            execute_on_edit=False,
            semantic_protocol=QA_VERIFIED_ANSWER_LINEAGE_PROTOCOL,
            recovery_policy="preserve_diagnose_repair_augment",
            required_evidence_tool_id=QA_RETRIEVAL_TOOL_ID,
        )
        for agent_id, passage_id in (
            ("reader", "initial-public"),
            ("repair_retriever", "repair-public"),
        ):
            env._progressive_outputs[agent_id] = (
                _test_evidence_retriever_artifact(passage_id)
            )
            env._progressive_output_metadata[agent_id] = {
                "tool_receipts": [_test_read_receipt(passage_id)]
            }
        env._failed_agent_ids.add("reasoner")
        env._react_exhausted_agent_ids.add("reasoner")
        env._repair_exhausted_agent_ids.add("reasoner")

        expected_route = {
            "source_id": "repair_retriever",
            "target_id": "reasoner",
            "source_to_target": True,
            "target_to_source": False,
        }
        self.assertIsNone(env.graph.output_agent_id)
        self.assertEqual(
            [],
            env._required_evidence_ingress_relation_candidates(),
        )
        self.assertEqual(
            [expected_route],
            env._repair_exhausted_relation_candidates(),
        )
        self.assertEqual(("set_relation",), env.model_admissible_action_types())
        self.assertEqual(
            [expected_route],
            env.model_admissible_action_targets()["set_relation"][
                "candidates"
            ],
        )

    def test_triviaqa_required_lineage_excludes_generic_output_bypass(
        self,
    ) -> None:
        registry = make_registry()
        runtime = _trivia_semantic_runtime(
            registry,
            _ImmediateGateway(),
        )
        env = AgentWorkflowEnv(
            registry,
            runtime=runtime,
            problem="What is the capital of France?",
            execute_on_edit=False,
            require_format_agent=True,
            semantic_protocol=QA_VERIFIED_ANSWER_LINEAGE_PROTOCOL,
            recovery_policy="preserve_diagnose_repair_augment",
            required_evidence_tool_id=QA_RETRIEVAL_TOOL_ID,
        )

        self.assertTrue(env._requires_complete_semantic_lineage())
        self.assertEqual(
            ("evidence_retriever", "reasoner", "verifier", "format"),
            env._missing_semantic_role_families(),
        )
        add_domain = env.model_admissible_action_targets()["add_subgraph"]
        self.assertTrue(add_domain["require_format_agent"])
        self.assertEqual("format", add_domain["output_role_family"])
        self.assertNotIn("output_role_families", add_domain)
        self.assertNotIn("output", add_domain["role_constraints"])
        self.assertEqual(
            {
                "execution_modes": ["react"],
                "allowed_tools": [[QA_RETRIEVAL_TOOL_ID]],
                "contract_responsibility": (
                    "ground answer-free evidence for the original entity and "
                    "requested relation in matching successful read Tool receipts"
                ),
            },
            add_domain["role_constraints"]["evidence_retriever"],
        )

        generic_output = AgentGraph(
            [
                AgentNode(
                    "output",
                    "cheap",
                    "retrieve and answer",
                    role_family="output",
                    allowed_tools=(QA_RETRIEVAL_TOOL_ID,),
                    execution_mode="react",
                )
            ],
            output_agent_id="output",
        )
        bypass_env = AgentWorkflowEnv(
            registry,
            runtime=runtime,
            graph=generic_output,
            problem="What is the capital of France?",
            execute_on_edit=False,
            require_format_agent=True,
            semantic_protocol=QA_VERIFIED_ANSWER_LINEAGE_PROTOCOL,
            recovery_policy="preserve_diagnose_repair_augment",
            required_evidence_tool_id=QA_RETRIEVAL_TOOL_ID,
        )
        bypass_execution = AgentRuntimeResult(
            run_id="generic-output-bypass",
            graph_revision=generic_output.revision,
            output_agent_id="output",
            final_answer="<answer>Paris</answer>",
            outputs={"output": "<answer>Paris</answer>"},
            output_metadata={
                "output": {
                    "tool_receipts": [_test_read_receipt("output-public")]
                }
            },
            calls=(),
            block_completion_order=(("output",),),
            executed_agent_ids=("output",),
        )
        issue = bypass_env._semantic_protocol_issue(bypass_execution)
        self.assertIn("Format Agent", issue or "")

    async def test_pending_isolated_retriever_repairs_before_another_add(
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
                "repair_retriever",
                "cheap",
                "retrieve evidence for the requested relation",
                role_family="evidence_retriever",
                allowed_tools=(QA_RETRIEVAL_TOOL_ID,),
                execution_mode="react",
                artifact_type="retrieval_evidence",
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
        env._latest_failure_record_by_agent["reasoner"] = (
            _react_exhaustion_record(
                graph,
                request_id="reasoner-pending-retriever",
            )
        )
        env._unresolved_dirty_agents.add("repair_retriever")

        self.assertEqual(
            ("repair_retriever",),
            env._pending_isolated_recovery_auxiliary_agent_ids(),
        )
        self.assertEqual(("modify_agent",), env.model_admissible_action_types())
        self.assertEqual(
            ["repair_retriever"],
            env.model_admissible_action_targets()["modify_agent"]["agent_ids"],
        )

        observed_execution: list[tuple[tuple[str, ...], str | None]] = []

        async def isolated_success(
            candidate_graph: AgentGraph,
            *args: object,
            **kwargs: object,
        ) -> AgentRuntimeResult:
            del args
            observed_execution.append(
                (
                    tuple(node.id for node in candidate_graph.nodes),
                    candidate_graph.output_agent_id,
                )
            )
            self.assertEqual({"repair_retriever"}, kwargs["dirty_agents"])
            return AgentRuntimeResult(
                run_id="pending-isolated-retriever-success",
                graph_revision=candidate_graph.revision,
                output_agent_id=None,
                final_answer=None,
                outputs={
                    "repair_retriever": _test_evidence_retriever_artifact(
                        "repair-public"
                    )
                },
                output_metadata={
                    "repair_retriever": {
                        "tool_receipts": [
                            _test_read_receipt("repair-public")
                        ]
                    }
                },
                calls=(),
                block_completion_order=(("repair_retriever",),),
                executed_agent_ids=("repair_retriever",),
            )

        with patch.object(runtime, "execute", side_effect=isolated_success):
            repaired = await env.step(
                json.dumps(
                    {
                        "action": "modify_agent",
                        "agent_id": "repair_retriever",
                        "contract": (
                            "continue the bounded evidence retrieval from the "
                            "preserved public continuation"
                        ),
                    }
                )
            )

        self.assertTrue(repaired.accepted, repaired.feedback)
        self.assertEqual([(("repair_retriever",), None)], observed_execution)
        self.assertEqual(("set_relation",), env.model_admissible_action_types())

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
        env._progressive_outputs["reader"] = _test_evidence_retriever_artifact(
            "reader-public"
        )
        env._progressive_output_metadata["reader"] = {
            "tool_receipts": [_test_read_receipt("reader-public")]
        }
        env._failed_agent_ids.update({"failed_reader", "reasoner"})
        env._repair_exhausted_agent_ids.update({"failed_reader", "reasoner"})
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
        env._latest_failure_record_by_agent["failed_reader"] = (
            _test_retrieval_failure_record(
                graph,
                agent_id="failed_reader",
                strategies=("initial_retrieval",),
                passage_ids=("new-public-read",),
            )
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
        env._progressive_outputs["reader"] = _test_evidence_retriever_artifact(
            "reader-public"
        )
        env._progressive_output_metadata["reader"] = {
            "tool_receipts": [_test_read_receipt("reader-public")]
        }
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

    async def test_exhausted_semantic_recovery_closes_generic_relation_and_modify_domains(
        self,
    ) -> None:
        graph = _trivia_semantic_graph()
        registry = make_registry()
        env = AgentWorkflowEnv(
            registry,
            runtime=_trivia_semantic_runtime(registry, _ImmediateGateway()),
            graph=graph,
            problem="Who wrote the novel?",
            execute_on_edit=False,
            max_agents=5,
            semantic_protocol=QA_VERIFIED_ANSWER_LINEAGE_PROTOCOL,
            recovery_policy="preserve_diagnose_repair_augment",
            required_evidence_tool_id=QA_RETRIEVAL_TOOL_ID,
        )
        env._failed_agent_ids.add("reasoner")
        env._repair_exhausted_agent_ids.add("reasoner")
        added = await env.step(
            json.dumps(
                {
                    "action": "add_subgraph",
                    "agents": [
                        {
                            "agent_id": "exhausted_reader",
                            "model_id": "cheap",
                            "contract": (
                                _QA_EVIDENCE_RETRIEVER_RECOVERY_CONTRACT
                            ),
                            "completion_condition": (
                                _QA_EVIDENCE_RETRIEVER_RECOVERY_COMPLETION
                            ),
                            "role_family": "evidence_retriever",
                            "allowed_tools": [QA_RETRIEVAL_TOOL_ID],
                            "execution_mode": "react",
                            "artifact_type": "retrieval_evidence",
                        }
                    ],
                    "relations": [],
                    "output_agent_id": None,
                }
            )
        )
        self.assertTrue(added.accepted, added.feedback)
        env._failed_agent_ids.update({"reasoner", "exhausted_reader"})
        env._repair_exhausted_agent_ids.update(
            {"reasoner", "exhausted_reader"}
        )
        env._unresolved_dirty_agents.update(
            {"reasoner", "verifier", "formatter", "exhausted_reader"}
        )

        generic_candidates = env._all_model_admissible_relation_candidates()
        self.assertTrue(generic_candidates)
        self.assertEqual([], env._model_admissible_relation_candidates())
        self.assertEqual((), env._model_admissible_modify_agent_ids())
        action_types = env.model_admissible_action_types()
        self.assertNotIn("set_relation", action_types)
        self.assertNotIn("modify_agent", action_types)
        targets = env.model_admissible_action_targets()
        self.assertNotIn("set_relation", targets)
        self.assertNotIn("modify_agent", targets)

        rejected = await env.step(
            json.dumps({"action": "set_relation", **generic_candidates[0]})
        )
        self.assertFalse(rejected.accepted)
        self.assertIn("artifact-free auxiliary", rejected.feedback)

    def test_modify_domain_targets_are_responsible_for_measured_failure(
        self,
    ) -> None:
        graph = _trivia_semantic_graph()
        registry = make_registry()
        env = AgentWorkflowEnv(
            registry,
            runtime=_trivia_semantic_runtime(registry, _ImmediateGateway()),
            graph=graph,
            problem="Who wrote the novel?",
            execute_on_edit=False,
            semantic_protocol=QA_VERIFIED_ANSWER_LINEAGE_PROTOCOL,
            recovery_policy="preserve_diagnose_repair_augment",
            required_evidence_tool_id=QA_RETRIEVAL_TOOL_ID,
        )
        env._failed_agent_ids.add("reasoner")

        modify_domain = env.model_admissible_action_targets()["modify_agent"]

        self.assertTrue(modify_domain["agent_ids"])
        self.assertLessEqual(
            set(modify_domain["agent_ids"]),
            set(modify_domain["responsible_agent_ids"]),
        )

    def test_modify_domain_omits_model_field_without_alternative_model(
        self,
    ) -> None:
        provider = ProviderSpec("only-provider", kind="test")
        registry = ModelRegistry(
            [provider],
            [ModelSpec("only-model", "only-provider")],
        )
        env = AgentWorkflowEnv(
            registry,
            _ImmediateGateway(),
            graph=AgentGraph(
                [AgentNode("worker", "only-model", "answer the task")]
            ),
            problem="question",
            execute_on_edit=False,
        )

        modify_domain = env.model_admissible_action_targets()["modify_agent"]
        candidate = modify_domain["per_agent_candidates"][0]

        self.assertNotIn("model_id", modify_domain["mutable_fields"])
        self.assertNotIn("model_id", candidate["mutable_fields"])
        self.assertNotIn(
            "model_id",
            candidate["discrete_value_domains"],
        )
        self.assertIn("contract", candidate["mutable_fields"])

    async def test_provider_failure_without_catalog_alternative_is_typed_terminal_and_reset_local(
        self,
    ) -> None:
        provider = ProviderSpec("only-provider", kind="test")
        registry = ModelRegistry(
            [provider],
            [ModelSpec("only-model", "only-provider")],
        )
        graph = AgentGraph(
            [AgentNode("worker", "only-model", "answer the task")]
        )
        env = AgentWorkflowEnv(
            registry,
            _ImmediateGateway(),
            graph=graph,
            problem="first question",
            execute_on_edit=False,
            recovery_policy="preserve_diagnose_repair_augment",
        )
        record = AgentFailureRecord(
            request_id="single-model-403",
            agent_id="worker",
            phase=ExecutionPhase.SINGLE,
            graph_revision=graph.revision,
            error_type="OpenAICompatibleGatewayError",
            message="provider request failed with HTTP status 403",
        )
        env._record_failure_state(
            (record,),
            current_agent_ids={"worker"},
        )

        self.assertEqual((), env.model_admissible_action_types())
        self.assertNotIn("modify_agent", env.model_admissible_action_targets())
        feedback = json.loads(
            env._execution_error_feedback(
                AgentRuntimeError("gateway failed", failure_records=(record,))
            ).split("=", 1)[1]
        )
        failed_agent = feedback["failed_agents"][0]
        self.assertNotIn("preferred_repair", failed_agent)
        self.assertEqual([], failed_agent["admitted_model_ids"])
        revision = env.revision

        rejected = await env.step(
            json.dumps(
                {
                    "action": "modify_agent",
                    "agent_id": "worker",
                    "contract": "change the task contract",
                }
            )
        )

        self.assertFalse(rejected.accepted)
        self.assertIn("no catalog-backed alternative", rejected.feedback)
        self.assertEqual(revision, env.revision)
        self.assertEqual("answer the task", env.graph.get_node("worker").contract)

        env.reset("second question", graph=graph)

        self.assertEqual([], env.model_availability_receipt()["unavailable_model_ids"])
        self.assertEqual([], env.model_availability_receipt()["failure_receipts"])
        self.assertIn("modify_agent", env.model_admissible_action_types())

    def test_triviaqa_partial_canvas_keeps_dedicated_retriever_optional(
        self,
    ) -> None:
        complete = _trivia_semantic_graph()
        graph = AgentGraph(
            [node for node in complete.nodes if node.id != "reader"],
            [
                AgentRelation("reasoner", "verifier", True, False),
                AgentRelation("verifier", "formatter", True, False),
            ],
            output_agent_id="formatter",
        )
        registry = make_registry()
        env = AgentWorkflowEnv(
            registry,
            runtime=_trivia_semantic_runtime(registry, _ImmediateGateway()),
            graph=graph,
            problem="What is the capital of France?",
            execute_on_edit=False,
            semantic_protocol=QA_VERIFIED_ANSWER_LINEAGE_PROTOCOL,
            recovery_policy="preserve_diagnose_repair_augment",
            required_evidence_tool_id=QA_RETRIEVAL_TOOL_ID,
        )

        self.assertEqual((), env._missing_semantic_role_families())
        self.assertIn("add_subgraph", env.model_admissible_action_types())
        domain = env.model_admissible_action_targets()["add_subgraph"]
        self.assertIn(
            "evidence_retriever",
            domain["admitted_new_role_families"],
        )
        self.assertNotEqual(
            ["evidence_retriever"],
            domain["admitted_new_role_families"],
        )

    async def test_triviaqa_failed_retriever_repairs_or_replaces_before_reasoner(
        self,
    ) -> None:
        complete = _trivia_semantic_graph()
        graph = AgentGraph([complete.get_node("reader")])
        registry = make_registry()
        env = AgentWorkflowEnv(
            registry,
            runtime=_trivia_semantic_runtime(registry, _ImmediateGateway()),
            graph=graph,
            problem="What is the capital of France?",
            execute_on_edit=False,
            semantic_protocol=QA_VERIFIED_ANSWER_LINEAGE_PROTOCOL,
            recovery_policy="preserve_diagnose_repair_augment",
            required_evidence_tool_id=QA_RETRIEVAL_TOOL_ID,
            require_format_agent=True,
        )
        record = AgentFailureRecord(
            request_id="reader-structured-artifact-exhaustion",
            agent_id="reader",
            phase=ExecutionPhase.SINGLE,
            graph_revision=graph.revision,
            error_type="ReactExecutionError",
            message="react agent 'reader' exhausted 20 turns",
            metadata={
                "react_trace": [
                    {
                        "turn": 20,
                        "observation_status": "schema_invalid",
                        "public_error_code": (
                            "qa_semantic_artifact_invalid: Evidence Retriever "
                            "entity_identity.evidence_surface does not occur "
                            "in evidence_span"
                        ),
                        "repair_instruction": (
                            "Preserve the same successful read receipt and "
                            "repair only the structured evidence artifact."
                        ),
                    }
                ],
                "tool_receipts": [_test_read_receipt("p1")],
                "tool_plan_exhausted": False,
            },
        )
        env._record_failure_state(
            (record,),
            current_agent_ids={"reader"},
        )

        modify_domain = env.model_admissible_action_targets()["modify_agent"]
        candidate = modify_domain["per_agent_candidates"][0]
        self.assertEqual(
            [_QA_EVIDENCE_RETRIEVER_RECOVERY_CONTRACT],
            candidate["discrete_value_domains"]["contract"],
        )
        self.assertEqual(
            [_QA_EVIDENCE_RETRIEVER_RECOVERY_COMPLETION],
            candidate["discrete_value_domains"]["completion_condition"],
        )
        wrong_completion = await env.step(
            '{"action":"modify_agent","agent_id":"reader",'
            '"completion_condition":"provides exact answer"}'
        )
        self.assertFalse(wrong_completion.accepted)
        self.assertIn(
            "preserve its evidence-only responsibility",
            wrong_completion.feedback,
        )

        # One same-Agent repair has now been measured as exhausted without a
        # new receipt.  The next executable unit must preserve and hand off the
        # existing public read, not materialize a blocked downstream role.
        env._repair_exhausted_agent_ids.add("reader")
        self.assertEqual(("add_subgraph",), env.model_admissible_action_types())
        add_domain = env.model_admissible_action_targets()["add_subgraph"]
        self.assertEqual(
            ["evidence_retriever"],
            add_domain["admitted_new_role_families"],
        )
        self.assertEqual([], add_domain["relations"])
        self.assertIsNone(add_domain["output_agent_id"])

        premature_reasoner = await env.step(
            json.dumps(
                {
                    "action": "add_subgraph",
                    "agents": [
                        {
                            "agent_id": "reasoner",
                            "model_id": "balanced",
                            "contract": (
                                "bind grounded evidence to the original answer "
                                "slot and requested relation, then derive one "
                                "semantic candidate"
                            ),
                            "role_family": "reasoner",
                            "allowed_tools": [QA_RETRIEVAL_TOOL_ID],
                            "execution_mode": "react",
                        }
                    ],
                    "relations": [],
                }
            )
        )
        self.assertFalse(premature_reasoner.accepted)
        self.assertIn(
            "same-role/same-artifact replacement",
            premature_reasoner.feedback,
        )

        replacement = await env.step(
            json.dumps(
                {
                    "action": "add_subgraph",
                    "agents": [
                        {
                            "agent_id": "replacement_reader",
                            "model_id": "cheap",
                            "contract": _QA_EVIDENCE_RETRIEVER_RECOVERY_CONTRACT,
                            "completion_condition": (
                                _QA_EVIDENCE_RETRIEVER_RECOVERY_COMPLETION
                            ),
                            "role_family": "evidence_retriever",
                            "allowed_tools": [QA_RETRIEVAL_TOOL_ID],
                            "execution_mode": "react",
                            "artifact_type": "retrieval_evidence",
                        }
                    ],
                    "relations": [],
                }
            )
        )
        self.assertTrue(replacement.accepted, replacement.feedback)

    def test_react_turn_exhausted_retriever_continuation_exposes_replacement(
        self,
    ) -> None:
        complete = _trivia_semantic_graph()
        graph = AgentGraph(
            [complete.get_node("reasoner"), complete.get_node("reader")],
            [AgentRelation("reader", "reasoner", True, False)],
        )
        registry = make_registry()
        env = AgentWorkflowEnv(
            registry,
            runtime=_trivia_semantic_runtime(registry, _ImmediateGateway()),
            graph=graph,
            problem=(
                "In which decade did Billboard magazine first publish an "
                "American hit chart?"
            ),
            execute_on_edit=False,
            max_agents=8,
            semantic_protocol=QA_VERIFIED_ANSWER_LINEAGE_PROTOCOL,
            recovery_policy="preserve_diagnose_repair_augment",
            required_evidence_tool_id=QA_RETRIEVAL_TOOL_ID,
        )
        record = _test_retrieval_turn_exhaustion_record(
            graph,
            agent_id="reader",
        )
        env._failed_agent_ids.add("reader")
        env._repair_exhausted_agent_ids.add("reader")
        env._latest_failure_record_by_agent["reader"] = record

        self.assertEqual(
            "react_turn_exhaustion",
            env._execution_failure_diagnosis(record)[0],
        )
        self.assertEqual(
            {"evidence_retriever": ("retrieval_evidence",)},
            env._repair_exhausted_auxiliary_replacement_domains(),
        )
        self.assertEqual(("add_subgraph",), env.model_admissible_action_types())
        add_domain = env.model_admissible_action_targets()["add_subgraph"]
        self.assertEqual(1, add_domain["min_new_agents"])
        self.assertEqual(1, add_domain["max_new_agents"])
        self.assertEqual(
            ["evidence_retriever"],
            add_domain["admitted_new_role_families"],
        )
        self.assertEqual([], add_domain["relations"])
        self.assertIsNone(add_domain["output_agent_id"])
        self.assertEqual(
            ["retrieval_evidence"],
            add_domain["role_constraints"]["evidence_retriever"][
                "artifact_types"
            ],
        )
        recovery = env.recovery_state()
        self.assertEqual("augment", recovery["phase"])
        self.assertEqual(["add_subgraph"], recovery["preferred_actions"])

    def test_react_turn_exhausted_retriever_terminal_receipts_close_replacement(
        self,
    ) -> None:
        cases = (
            {"continuation_admissible": False},
            {"tool_plan_exhausted": True},
            {"bounded_schedule_exhausted": True},
            {"remaining_tool_calls": 0},
            {"strategies": (), "passage_ids": ()},
        )
        for overrides in cases:
            with self.subTest(overrides=overrides):
                complete = _trivia_semantic_graph()
                graph = AgentGraph(
                    [
                        complete.get_node("reasoner"),
                        complete.get_node("reader"),
                    ],
                    [AgentRelation("reader", "reasoner", True, False)],
                )
                registry = make_registry()
                env = AgentWorkflowEnv(
                    registry,
                    runtime=_trivia_semantic_runtime(
                        registry,
                        _ImmediateGateway(),
                    ),
                    graph=graph,
                    problem="Who is the requested entity?",
                    execute_on_edit=False,
                    max_agents=8,
                    semantic_protocol=QA_VERIFIED_ANSWER_LINEAGE_PROTOCOL,
                    recovery_policy="preserve_diagnose_repair_augment",
                    required_evidence_tool_id=QA_RETRIEVAL_TOOL_ID,
                )
                record = _test_retrieval_turn_exhaustion_record(
                    graph,
                    agent_id="reader",
                    **overrides,
                )
                env._failed_agent_ids.add("reader")
                env._repair_exhausted_agent_ids.add("reader")
                env._latest_failure_record_by_agent["reader"] = record

                self.assertEqual(
                    {},
                    env._repair_exhausted_auxiliary_replacement_domains(),
                )
                self.assertEqual((), env.model_admissible_action_types())
                self.assertEqual({}, env.model_admissible_action_targets())
                self.assertEqual([], env.recovery_state()["preferred_actions"])

    def test_react_turn_exhausted_replacement_requires_new_public_progress(
        self,
    ) -> None:
        for replacement_strategies, expected in (
            (("initial_retrieval",), {}),
            (
                ("initial_retrieval", "alias_expansion"),
                {"evidence_retriever": ("retrieval_evidence",)},
            ),
        ):
            with self.subTest(
                replacement_strategies=replacement_strategies
            ):
                complete = _trivia_semantic_graph()
                graph = AgentGraph(
                    [
                        complete.get_node("reasoner"),
                        complete.get_node("reader"),
                        AgentNode(
                            "replacement_reader",
                            "cheap",
                            _QA_EVIDENCE_RETRIEVER_RECOVERY_CONTRACT,
                            role_family="evidence_retriever",
                            allowed_tools=(QA_RETRIEVAL_TOOL_ID,),
                            execution_mode="react",
                            artifact_type="retrieval_evidence",
                            completion_condition=(
                                _QA_EVIDENCE_RETRIEVER_RECOVERY_COMPLETION
                            ),
                        ),
                    ],
                    [
                        AgentRelation("reader", "reasoner", True, False),
                        AgentRelation(
                            "replacement_reader",
                            "reasoner",
                            True,
                            False,
                        ),
                    ],
                )
                registry = make_registry()
                env = AgentWorkflowEnv(
                    registry,
                    runtime=_trivia_semantic_runtime(
                        registry,
                        _ImmediateGateway(),
                    ),
                    graph=graph,
                    problem="Who is the requested entity?",
                    execute_on_edit=False,
                    max_agents=8,
                    semantic_protocol=QA_VERIFIED_ANSWER_LINEAGE_PROTOCOL,
                    recovery_policy="preserve_diagnose_repair_augment",
                    required_evidence_tool_id=QA_RETRIEVAL_TOOL_ID,
                )
                env._failed_agent_ids.update(
                    {"reader", "replacement_reader"}
                )
                env._repair_exhausted_agent_ids.update(
                    {"reader", "replacement_reader"}
                )
                env._latest_failure_record_by_agent.update(
                    {
                        "reader": _test_retrieval_turn_exhaustion_record(
                            graph,
                            agent_id="reader",
                            strategies=("initial_retrieval",),
                            passage_ids=("shared-public-read",),
                        ),
                        "replacement_reader": (
                            _test_retrieval_turn_exhaustion_record(
                                graph,
                                agent_id="replacement_reader",
                                strategies=replacement_strategies,
                                passage_ids=("shared-public-read",),
                            )
                        ),
                    }
                )

                self.assertEqual(
                    expected,
                    env._repair_exhausted_auxiliary_replacement_domains(),
                )
                expected_actions = ("add_subgraph",) if expected else ()
                self.assertEqual(
                    expected_actions,
                    env.model_admissible_action_types(),
                )
                self.assertEqual(
                    ["add_subgraph"] if expected else [],
                    env.recovery_state()["preferred_actions"],
                )

    def test_react_turn_exhausted_replacement_counts_verified_recall_expansion(
        self,
    ) -> None:
        complete = _trivia_semantic_graph()
        graph = AgentGraph(
            [
                complete.get_node("reasoner"),
                complete.get_node("reader"),
                AgentNode(
                    "replacement_reader",
                    "cheap",
                    _QA_EVIDENCE_RETRIEVER_RECOVERY_CONTRACT,
                    role_family="evidence_retriever",
                    allowed_tools=(QA_RETRIEVAL_TOOL_ID,),
                    execution_mode="react",
                    artifact_type="retrieval_evidence",
                    completion_condition=(
                        _QA_EVIDENCE_RETRIEVER_RECOVERY_COMPLETION
                    ),
                ),
            ],
            [
                AgentRelation("reader", "reasoner", True, False),
                AgentRelation(
                    "replacement_reader",
                    "reasoner",
                    True,
                    False,
                ),
            ],
        )
        registry = make_registry()
        env = AgentWorkflowEnv(
            registry,
            runtime=_trivia_semantic_runtime(registry, _ImmediateGateway()),
            graph=graph,
            problem="Who is the requested entity?",
            execute_on_edit=False,
            max_agents=8,
            semantic_protocol=QA_VERIFIED_ANSWER_LINEAGE_PROTOCOL,
            recovery_policy="preserve_diagnose_repair_augment",
            required_evidence_tool_id=QA_RETRIEVAL_TOOL_ID,
        )
        verified_recall = {
            "verified": True,
            "recall_expansion": True,
            "fts_term_set": ["billboard", "chart", "publish"],
            "observed_top_k": 20,
        }
        env._failed_agent_ids.update({"reader", "replacement_reader"})
        env._repair_exhausted_agent_ids.update(
            {"reader", "replacement_reader"}
        )
        env._latest_failure_record_by_agent.update(
            {
                "reader": _test_retrieval_turn_exhaustion_record(
                    graph,
                    agent_id="reader",
                    strategies=("initial_retrieval", "alias_expansion"),
                    passage_ids=(),
                ),
                "replacement_reader": _test_retrieval_turn_exhaustion_record(
                    graph,
                    agent_id="replacement_reader",
                    strategies=("initial_retrieval", "alias_expansion"),
                    passage_ids=(),
                    retrieval_attempts=(verified_recall,),
                ),
            }
        )

        self.assertEqual(
            {"evidence_retriever": ("retrieval_evidence",)},
            env._repair_exhausted_auxiliary_replacement_domains(),
        )

        env._latest_failure_record_by_agent["reader"] = (
            _test_retrieval_turn_exhaustion_record(
                graph,
                agent_id="reader",
                strategies=("initial_retrieval", "alias_expansion"),
                passage_ids=(),
                retrieval_attempts=(verified_recall,),
            )
        )
        self.assertEqual(
            {},
            env._repair_exhausted_auxiliary_replacement_domains(),
        )

    async def test_triviaqa_valid_retriever_artifact_admits_any_direct_reasoner_ingress(
        self,
    ) -> None:
        complete = _trivia_semantic_graph(reader_to_reasoner=False)
        graph = AgentGraph(
            [node for node in complete.nodes if node.id != "formatter"],
            [AgentRelation("reasoner", "verifier", True, False)],
        )
        graph.add_agent(
            AgentNode(
                "reader_b",
                "fast",
                "retrieve an independent answer-free evidence artifact",
                role_family="evidence_retriever",
                allowed_tools=(QA_RETRIEVAL_TOOL_ID,),
                execution_mode="react",
                artifact_type="retrieval_evidence",
            )
        )
        registry = make_registry()
        env = AgentWorkflowEnv(
            registry,
            runtime=_trivia_semantic_runtime(registry, _ImmediateGateway()),
            graph=graph,
            problem="What is the capital of France?",
            execute_on_edit=False,
            semantic_protocol=QA_VERIFIED_ANSWER_LINEAGE_PROTOCOL,
            recovery_policy="preserve_diagnose_repair_augment",
            required_evidence_tool_id=QA_RETRIEVAL_TOOL_ID,
        )
        env._progressive_outputs["reader"] = _test_evidence_retriever_artifact(
            "p1"
        )
        env._progressive_output_metadata["reader"] = {
            "tool_receipts": [_test_read_receipt("p1")],
            "artifact_version": "fixture:reader:current",
        }
        env._progressive_outputs["reader_b"] = _test_evidence_retriever_artifact(
            "p1"
        )
        env._progressive_output_metadata["reader_b"] = {
            "tool_receipts": [_test_read_receipt("p1")],
            "artifact_version": "fixture:reader-b:current",
        }

        self.assertEqual(("set_relation",), env.model_admissible_action_types())
        candidates = env.model_admissible_action_targets()["set_relation"][
            "candidates"
        ]
        self.assertIn(
            {
                "source_id": "reader",
                "target_id": "reasoner",
                "source_to_target": True,
                "target_to_source": False,
            },
            candidates,
        )
        self.assertIn(
            {
                "source_id": "reader_b",
                "target_id": "reasoner",
                "source_to_target": True,
                "target_to_source": False,
            },
            candidates,
        )
        revision = env.revision

        rejected = await env.step(
            json.dumps(
                {
                    "action": "set_relation",
                    "source_id": "reader",
                    "target_id": "reasoner",
                    "source_to_target": False,
                    "target_to_source": True,
                }
            )
        )

        self.assertFalse(rejected.accepted)
        self.assertIn("exact admitted set_relation candidate", rejected.feedback)
        self.assertEqual(revision, env.graph.revision)

    async def test_triviaqa_reciprocal_runtime_keeps_four_phase_causal_order(
        self,
    ) -> None:
        graph = _trivia_semantic_graph()
        graph.set_relation("reader", "reasoner", True, True)
        registry = make_registry()
        gateway = _ImmediateGateway()
        env = AgentWorkflowEnv(
            registry,
            runtime=_trivia_semantic_runtime(registry, gateway),
            graph=graph,
            problem="What is the capital of France?",
            execute_on_edit=True,
            require_format_agent=True,
            semantic_protocol=QA_VERIFIED_ANSWER_LINEAGE_PROTOCOL,
            recovery_policy="preserve_diagnose_repair_augment",
            required_evidence_tool_id=QA_RETRIEVAL_TOOL_ID,
        )

        executed = await env.step(
            '{"action":"modify_agent","agent_id":"reader",'
            '"contract":"retrieve answer-free public evidence for the original '
            'entity and requested relation"}'
        )

        self.assertTrue(executed.accepted, executed.feedback)
        self.assertIsNotNone(executed.execution)
        assert executed.execution is not None
        self.assertEqual(
            [
                "reader",
                "reasoner",
                "reader",
                "reasoner",
                "verifier",
                "formatter",
            ],
            [item.agent.id for item in gateway.requests],
        )
        self.assertEqual((), executed.execution.deferred_agent_ids)
        # AgentRuntime owns the existing Retriever DRAFT -> Reasoner DRAFT ->
        # Retriever REVISION -> Reasoner REVISION causal block.  Env cache
        # admission must not rewrite the Director-selected reciprocal Canvas.
        self.assertTrue(
            env.graph.relation_bits("reader", "reasoner").is_bidirectional
        )

    async def test_triviaqa_relation_mismatched_cache_reexecutes_before_reasoner(
        self,
    ) -> None:
        graph = _trivia_semantic_graph()
        registry = make_registry()
        gateway = _ImmediateGateway()
        env = AgentWorkflowEnv(
            registry,
            runtime=_trivia_semantic_runtime(registry, gateway),
            graph=graph,
            problem="What is the capital of France?",
            execute_on_edit=True,
            require_format_agent=True,
            semantic_protocol=QA_VERIFIED_ANSWER_LINEAGE_PROTOCOL,
            recovery_policy="preserve_diagnose_repair_augment",
            required_evidence_tool_id=QA_RETRIEVAL_TOOL_ID,
        )
        mismatched = json.loads(_test_evidence_retriever_artifact("p1"))
        mismatched["target_relation"] = "birth place"
        mismatched["evidence_proposition"]["predicate"] = "was born in"
        env._progressive_outputs["reader"] = json.dumps(mismatched)
        env._progressive_output_metadata["reader"] = {
            "artifact_version": "reader:current",
            "tool_receipts": [_test_read_receipt("p1")],
        }
        self.assertFalse(
            env._triviaqa_retriever_has_current_grounded_artifact(
                env.graph,
                "reader",
            )
        )

        executed = await env.step(
            '{"action":"modify_agent","agent_id":"reasoner",'
            '"contract":"bind grounded propositions to the original requested '
            'answer slot and relation, then derive one semantic candidate"}'
        )

        self.assertTrue(executed.accepted, executed.feedback)
        self.assertIsNotNone(executed.execution)
        assert executed.execution is not None
        request_ids = [item.agent.id for item in gateway.requests]
        self.assertEqual("reader", request_ids[0])
        self.assertEqual("reasoner", request_ids[1])
        self.assertNotIn("reader", executed.execution.reused_agent_ids)
        self.assertNotIn("reasoner", executed.execution.deferred_agent_ids)

    async def test_triviaqa_fan_in_reexecutes_only_invalid_retriever_cache(
        self,
    ) -> None:
        graph = _trivia_semantic_graph()
        graph.add_agent(
            AgentNode(
                "reader_b",
                "fast",
                "retrieve independent answer-free evidence for the question",
                role_family="evidence_retriever",
                allowed_tools=(QA_RETRIEVAL_TOOL_ID,),
                execution_mode="react",
                artifact_type="retrieval_evidence",
            )
        )
        graph.set_relation("reader_b", "reasoner", True, False)
        registry = make_registry()
        gateway = _ImmediateGateway()
        env = AgentWorkflowEnv(
            registry,
            runtime=_trivia_semantic_runtime(registry, gateway),
            graph=graph,
            problem="What is the capital of France?",
            execute_on_edit=True,
            require_format_agent=True,
            semantic_protocol=QA_VERIFIED_ANSWER_LINEAGE_PROTOCOL,
            recovery_policy="preserve_diagnose_repair_augment",
            required_evidence_tool_id=QA_RETRIEVAL_TOOL_ID,
        )
        env._progressive_outputs["reader"] = (
            _test_evidence_retriever_artifact("valid-passage")
        )
        env._progressive_output_metadata["reader"] = {
            "artifact_version": "reader:current",
            "tool_receipts": [_test_read_receipt("valid-passage")],
        }
        mismatched = json.loads(
            _test_evidence_retriever_artifact("invalid-passage")
        )
        mismatched["target_relation"] = "birth place"
        mismatched["evidence_proposition"]["predicate"] = "was born in"
        env._progressive_outputs["reader_b"] = json.dumps(mismatched)
        env._progressive_output_metadata["reader_b"] = {
            "artifact_version": "reader-b:current",
            "tool_receipts": [_test_read_receipt("invalid-passage")],
        }

        self.assertEqual(
            {"reader_b"},
            env._triviaqa_retrievers_requiring_validation(env.graph),
        )
        executed = await env.step(
            '{"action":"modify_agent","agent_id":"reasoner",'
            '"contract":"bind grounded propositions to the original requested '
            'answer slot and relation, then derive one semantic candidate"}'
        )

        self.assertTrue(executed.accepted, executed.feedback)
        self.assertIsNotNone(executed.execution)
        assert executed.execution is not None
        request_ids = [item.agent.id for item in gateway.requests]
        self.assertEqual("reader_b", request_ids[0])
        self.assertEqual("reasoner", request_ids[1])
        self.assertIn("reader", executed.execution.reused_agent_ids)
        self.assertNotIn("reader_b", executed.execution.reused_agent_ids)

    async def test_triviaqa_evidence_ingress_mask_matches_preservation_admission_before_output(
        self,
    ) -> None:
        complete = _trivia_semantic_graph(reader_to_reasoner=False)
        graph = AgentGraph(
            [node for node in complete.nodes if node.id != "formatter"],
        )
        registry = make_registry()
        env = AgentWorkflowEnv(
            registry,
            runtime=_trivia_semantic_runtime(registry, _ImmediateGateway()),
            graph=graph,
            problem="What is the capital of France?",
            execute_on_edit=False,
            semantic_protocol=QA_VERIFIED_ANSWER_LINEAGE_PROTOCOL,
            recovery_policy="preserve_diagnose_repair_augment",
            required_evidence_tool_id=QA_RETRIEVAL_TOOL_ID,
        )
        env._progressive_outputs["reader"] = _test_evidence_retriever_artifact(
            "p1"
        )
        env._progressive_output_metadata["reader"] = {
            "tool_receipts": [_test_read_receipt("p1")],
            "artifact_version": "fixture:reader:current",
        }

        self.assertEqual(("set_relation",), env.model_admissible_action_types())
        self.assertEqual(
            [
                {
                    "source_id": "reader",
                    "target_id": "reasoner",
                    "source_to_target": True,
                    "target_to_source": False,
                }
            ],
            env.model_admissible_action_targets()["set_relation"]["candidates"],
        )

        routed = await env.step(
            '{"action":"set_relation","source_id":"reader",'
            '"target_id":"reasoner","source_to_target":true,'
            '"target_to_source":false}'
        )

        self.assertTrue(routed.accepted)
        self.assertTrue(
            env.graph.relation_bits("reader", "reasoner").source_to_target
        )
        self.assertIn("add_subgraph", env.model_admissible_action_types())
        add_domain = env.model_admissible_action_targets()["add_subgraph"]
        self.assertEqual(
            ["reader"],
            add_domain["preserved_input_agent_ids"],
        )
        self.assertIn(
            "format",
            add_domain["admitted_new_role_families"],
        )

    async def test_triviaqa_missing_retriever_add_domain_matches_canvas_admission(
        self,
    ) -> None:
        registry = make_registry()
        env = AgentWorkflowEnv(
            registry,
            runtime=_trivia_semantic_runtime(registry, _ImmediateGateway()),
            problem="What is the capital of France?",
            execute_on_edit=False,
            require_format_agent=True,
            semantic_protocol=QA_VERIFIED_ANSWER_LINEAGE_PROTOCOL,
            recovery_policy="preserve_diagnose_repair_augment",
            required_evidence_tool_id=QA_RETRIEVAL_TOOL_ID,
        )
        # Reconstruct the accepted partial revision captured in the tc_5 live
        # trajectory.  Constructor validation intentionally accepts only
        # complete Format ownership, while a progressive Canvas can persist
        # this deferred intermediate revision between edits.
        complete = _trivia_semantic_graph()
        env._graph = AgentGraph(
            [node for node in complete.nodes if node.id != "reader"],
            [AgentRelation("reasoner", "verifier", True, False)],
        )

        self.assertEqual(
            ("evidence_retriever",),
            env._missing_semantic_role_families(),
        )
        self.assertEqual(("add_subgraph",), env.model_admissible_action_types())
        self.assertEqual(
            ["evidence_retriever"],
            env.model_admissible_action_targets()["add_subgraph"][
                "admitted_new_role_families"
            ],
        )

        added = await env.step(
            json.dumps(
                {
                    "action": "add_subgraph",
                    "agents": [
                        {
                            "agent_id": "reader",
                            "model_id": "cheap",
                            "contract": (
                                "retrieve answer-free evidence for the question "
                                "entity and relation"
                            ),
                            "role_family": "evidence_retriever",
                            "allowed_tools": [QA_RETRIEVAL_TOOL_ID],
                            "execution_mode": "react",
                            "artifact_type": "retrieval_evidence",
                        }
                    ],
                    "relations": [
                        {
                            "source_id": "reader",
                            "target_id": "reasoner",
                            "source_to_target": True,
                            "target_to_source": False,
                        }
                    ],
                }
            )
        )

        self.assertTrue(added.accepted, added.feedback)
        self.assertEqual(("set_relation",), env.model_admissible_action_types())
        self.assertEqual(
            [
                {
                    "source_id": "verifier",
                    "target_id": "formatter",
                    "source_to_target": True,
                    "target_to_source": False,
                }
            ],
            env.model_admissible_action_targets()["set_relation"]["candidates"],
        )

    async def test_triviaqa_evidence_ingress_precedes_missing_formatter_add(
        self,
    ) -> None:
        complete = _trivia_semantic_graph(reader_to_reasoner=False)
        graph = AgentGraph(
            [node for node in complete.nodes if node.id != "formatter"],
        )
        registry = make_registry()
        env = AgentWorkflowEnv(
            registry,
            runtime=_trivia_semantic_runtime(registry, _ImmediateGateway()),
            graph=graph,
            problem="What is the capital of France?",
            execute_on_edit=False,
            require_format_agent=True,
            semantic_protocol=QA_VERIFIED_ANSWER_LINEAGE_PROTOCOL,
            recovery_policy="preserve_diagnose_repair_augment",
            required_evidence_tool_id=QA_RETRIEVAL_TOOL_ID,
        )
        env._progressive_outputs["reader"] = _test_evidence_retriever_artifact(
            "p1"
        )
        env._progressive_output_metadata["reader"] = {
            "tool_receipts": [_test_read_receipt("p1")],
            "artifact_version": "fixture:reader:current",
        }

        self.assertEqual(("format",), env._missing_semantic_role_families())
        actions = env.model_admissible_action_types()
        targets = env.model_admissible_action_targets()
        self.assertEqual(("set_relation",), actions)
        self.assertEqual(
            [
                {
                    "source_id": "reader",
                    "target_id": "reasoner",
                    "source_to_target": True,
                    "target_to_source": False,
                }
            ],
            targets["set_relation"]["candidates"],
        )
        director_validate_live_action_target_domains(actions, targets)

        rejected = await env.step(
            json.dumps(
                {
                    "action": "add_subgraph",
                    "agents": [
                        {
                            "agent_id": "formatter",
                            "model_id": "fast",
                            "contract": (
                                "copy the routed semantic candidate "
                                "character-for-character into the required "
                                "answer wrapper"
                            ),
                            "role_family": "format",
                            "allowed_tools": [],
                            "execution_mode": "reasoning",
                        }
                    ],
                    "relations": [],
                    "output_agent_id": "formatter",
                }
            )
        )

        self.assertFalse(rejected.accepted)
        self.assertIn("receipt-grounded Evidence Retriever", rejected.feedback)

    async def test_grounded_orphan_retriever_does_not_invalidate_verified_lineage(
        self,
    ) -> None:
        complete = _hotpot_semantic_graph()
        registry = make_registry()
        gateway = _HotpotSemanticGateway()
        runtime = _hotpot_semantic_runtime(registry, gateway)
        initial = await runtime.execute(
            complete,
            "What is the capital of France?",
            require_complete=True,
            format_output_agent=True,
        )
        graph = complete.fork()
        graph.set_relation("reader", "reasoner", False, False)
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
        env._progressive_outputs = dict(initial.outputs)
        env._progressive_output_metadata = {
            agent_id: dict(metadata)
            for agent_id, metadata in initial.output_metadata.items()
        }
        env._progressive_outputs["reader"] = _test_evidence_retriever_artifact(
            "p1"
        )
        # The active Reasoner obtained its own public read in the tc1 shape;
        # keep that receipt while the detached Retriever remains independently
        # grounded and terminal-unreachable.
        env._progressive_output_metadata["reasoner"] = {
            **env._progressive_output_metadata["reasoner"],
            "tool_receipts": list(
                env._progressive_output_metadata["reader"]["tool_receipts"]
            ),
            "input_artifact_versions": {},
        }
        current_outputs = {
            **dict(initial.outputs),
            "reader": _test_evidence_retriever_artifact("p1"),
        }
        current_metadata = {
            agent_id: dict(metadata)
            for agent_id, metadata in env._progressive_output_metadata.items()
        }
        current_execution = replace(
            initial,
            graph_revision=graph.revision,
            outputs=current_outputs,
            output_metadata=current_metadata,
        )
        env._progressive_outputs = current_outputs
        env._progressive_output_metadata = current_metadata
        env._progressive_execution = current_execution
        env._progressive_execution_revision = graph.revision

        self.assertEqual(
            ("reasoner", "verifier", "formatter"),
            env._active_semantic_lineage_ids(),
        )
        self.assertEqual(("reader",), env._terminal_unreachable_agent_ids())
        self.assertTrue(env.finish_admissibility()["admissible"])
        self.assertEqual(("finish",), env.model_admissible_action_types())
        request_count = len(gateway.requests)

        result = await env.step(json.dumps({"action": "finish"}))

        self.assertTrue(result.accepted, result.feedback)
        self.assertTrue(result.done)
        self.assertTrue(result.execution_reused)
        self.assertEqual("<answer>Paris</answer>", result.execution.final_answer)
        self.assertEqual(request_count, len(gateway.requests))
        self.assertNotIn(
            "reasoner",
            env._directed_successors(graph, "reader"),
        )
        self.assertEqual(
            _test_evidence_retriever_artifact("p1"),
            env._progressive_outputs["reader"],
        )

    async def test_orphan_retriever_remains_required_when_reasoner_lacks_read_receipt(
        self,
    ) -> None:
        complete = _hotpot_semantic_graph()
        registry = make_registry()
        runtime = _hotpot_semantic_runtime(registry, _HotpotSemanticGateway())
        initial = await runtime.execute(
            complete,
            "What is the capital of France?",
            require_complete=True,
            format_output_agent=True,
        )
        graph = complete.fork()
        graph.set_relation("reader", "reasoner", False, False)
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
        outputs = {
            **dict(initial.outputs),
            "reader": _test_evidence_retriever_artifact("p1"),
        }
        metadata = {
            agent_id: dict(value)
            for agent_id, value in initial.output_metadata.items()
        }
        metadata["reader"] = dict(
            initial.output_metadata["reader"]
        )
        metadata["reasoner"] = {
            **metadata["reasoner"],
            "tool_receipts": [],
            "input_artifact_versions": {},
        }
        env._progressive_outputs = outputs
        env._progressive_output_metadata = metadata
        env._progressive_execution = replace(
            initial,
            graph_revision=graph.revision,
            outputs=outputs,
            output_metadata=metadata,
        )
        env._progressive_execution_revision = graph.revision

        self.assertFalse(env.finish_admissibility()["admissible"])
        self.assertIn("set_relation", env.model_admissible_action_types())
        self.assertIn(
            {
                "source_id": "reader",
                "target_id": "reasoner",
                "source_to_target": True,
                "target_to_source": False,
            },
            env.model_admissible_action_targets()["set_relation"]["candidates"],
        )

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
        env._latest_failure_record_by_agent["failed_reader"] = (
            _test_retrieval_failure_record(
                graph,
                agent_id="failed_reader",
                strategies=("initial_retrieval",),
                passage_ids=("new-public-read",),
            )
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
        grounded_artifact = {
            "question_scope": "What is the capital of France?",
            "entity_identity": {
                "question_surface": "France",
                "evidence_surface": "France",
            },
            "target_relation": "capital of",
            "answer_type_constraint": "short_answer",
            "evidence_proposition": {
                "subject": "Paris",
                "predicate": "is the capital of",
                "object_or_attribute_value": "France",
            },
            "evidence_span": "Paris is the capital of France.",
            "passage_id": "replacement-public",
        }
        env._progressive_outputs["replacement_reader"] = json.dumps(
            grounded_artifact
        )
        self.assertFalse(
            env._semantic_replacement_has_valid_artifact(
                "replacement_reader",
                "evidence_retriever",
            )
        )
        env._progressive_output_metadata["replacement_reader"] = {
            "tool_receipts": [_test_read_receipt("replacement-public")]
        }

        relation_drift = json.loads(json.dumps(grounded_artifact))
        relation_drift["target_relation"] = "introduced"
        relation_drift["evidence_proposition"]["predicate"] = "introduced"
        env._progressive_outputs["replacement_reader"] = json.dumps(
            relation_drift
        )
        self.assertFalse(
            env._semantic_replacement_has_valid_artifact(
                "replacement_reader",
                "evidence_retriever",
            )
        )

        env._progressive_outputs["replacement_reader"] = json.dumps(
            grounded_artifact
        )
        self.assertTrue(
            env._semantic_replacement_has_valid_artifact(
                "replacement_reader",
                "evidence_retriever",
            )
        )
        env._failed_agent_ids.update({"failed_reader", "reasoner"})
        env._repair_exhausted_agent_ids.update({"failed_reader", "reasoner"})
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

    async def test_dirty_replacement_excludes_repeat_add_below_agent_capacity(
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
            # Six Agents are present, so capacity remains.  An active isolated
            # replacement must still own the next recovery edit instead of
            # admitting another replacement branch.
            max_agents=8,
            semantic_protocol="hotpotqa_verified_answer_slot_v1",
            recovery_policy="preserve_diagnose_repair_augment",
            required_evidence_tool_id=QA_RETRIEVAL_TOOL_ID,
        )
        env._failed_agent_ids.update({"failed_reader", "reasoner"})
        env._repair_exhausted_agent_ids.update({"failed_reader", "reasoner"})
        env._latest_failure_record_by_agent["failed_reader"] = (
            _test_retrieval_failure_record(
                graph,
                agent_id="failed_reader",
                strategies=("initial_retrieval",),
                passage_ids=("new-public-read",),
            )
        )
        env._unresolved_dirty_agents.update(
            {"replacement_reader", "reasoner", "verifier", "formatter"}
        )

        self.assertEqual(("modify_agent",), env.model_admissible_action_types())
        recovery = env.recovery_state()
        self.assertEqual("diagnose_repair", recovery["phase"])
        self.assertEqual(
            ["replacement_reader"],
            recovery["active_auxiliary_replacement_agent_ids"],
        )
        self.assertEqual(["modify_agent"], recovery["preferred_actions"])
        modify_domain = env.model_admissible_action_targets()["modify_agent"]
        self.assertEqual(
            ["replacement_reader"],
            modify_domain["agent_ids"],
        )
        self.assertLessEqual(
            set(modify_domain["agent_ids"]),
            set(modify_domain["responsible_agent_ids"]),
        )
        self.assertEqual(
            ["contract", "completion_condition"],
            modify_domain["mutable_fields"],
        )
        self.assertEqual(
            ["contract", "completion_condition"],
            modify_domain["per_agent_candidates"][0]["mutable_fields"],
        )
        self.assertEqual(
            {
                "contract": "retrieve replacement evidence",
                "completion_condition": None,
            },
            modify_domain["per_agent_candidates"][0]["current_values"],
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
        self.assertIn(
            "before augmentation or modification of blocked downstream Agents",
            rejected.feedback,
        )
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

    async def test_repair_exhausted_replacement_with_new_progress_exposes_add(
        self,
    ) -> None:
        graph = _trivia_semantic_graph()
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
            runtime=_trivia_semantic_runtime(registry, _ImmediateGateway()),
            graph=graph,
            problem="Who wrote the novel?",
            execute_on_edit=False,
            max_agents=8,
            semantic_protocol=QA_VERIFIED_ANSWER_LINEAGE_PROTOCOL,
            recovery_policy="preserve_diagnose_repair_augment",
            required_evidence_tool_id=QA_RETRIEVAL_TOOL_ID,
        )
        env._failed_agent_ids.update({"failed_reader", "reasoner"})
        env._repair_exhausted_agent_ids.update(
            {"failed_reader", "reasoner"}
        )
        env._latest_failure_record_by_agent["failed_reader"] = (
            _test_retrieval_failure_record(
                graph,
                agent_id="failed_reader",
                strategies=("initial_retrieval",),
                passage_ids=("public-1",),
            )
        )
        added = await env.step(
            json.dumps(
                {
                    "action": "add_subgraph",
                    "agents": [
                        {
                            "agent_id": "replacement_reader",
                            "model_id": "cheap",
                            "contract": (
                                _QA_EVIDENCE_RETRIEVER_RECOVERY_CONTRACT
                            ),
                            "completion_condition": (
                                _QA_EVIDENCE_RETRIEVER_RECOVERY_COMPLETION
                            ),
                            "role_family": "evidence_retriever",
                            "allowed_tools": [QA_RETRIEVAL_TOOL_ID],
                            "execution_mode": "react",
                            "artifact_type": "retrieval_evidence",
                        }
                    ],
                    "relations": [],
                    "output_agent_id": None,
                }
            )
        )
        self.assertTrue(added.accepted, added.feedback)
        env._failed_agent_ids.update(
            {"failed_reader", "replacement_reader", "reasoner"}
        )
        env._repair_exhausted_agent_ids.update(
            {"failed_reader", "replacement_reader", "reasoner"}
        )
        env._latest_failure_record_by_agent.update(
            {
                "replacement_reader": _test_retrieval_failure_record(
                    env.graph,
                    agent_id="replacement_reader",
                    strategies=(
                        "initial_retrieval",
                        "spelling_normalization",
                    ),
                    passage_ids=("public-1", "public-2"),
                ),
            }
        )
        env._unresolved_dirty_agents.update(
            {"replacement_reader", "reasoner", "verifier", "formatter"}
        )

        self.assertEqual((), env._dirty_auxiliary_replacement_agent_ids())
        self.assertEqual(("add_subgraph",), env.model_admissible_action_types())
        recovery = env.recovery_state()
        self.assertEqual([], recovery["active_auxiliary_replacement_agent_ids"])
        add_domain = env.model_admissible_action_targets()["add_subgraph"]
        self.assertEqual(1, add_domain["min_new_agents"])
        self.assertEqual(1, add_domain["max_new_agents"])
        self.assertEqual(
            ["evidence_retriever"],
            add_domain["admitted_new_role_families"],
        )
        self.assertEqual([], add_domain["relations"])
        self.assertIsNone(add_domain["output_agent_id"])
        self.assertEqual(
            ["retrieval_evidence"],
            add_domain["role_constraints"]["evidence_retriever"][
                "artifact_types"
            ],
        )

        rejected_modify = await env.step(
            '{"action":"modify_agent","agent_id":"replacement_reader",'
            '"contract":"retry the exhausted replacement"}'
        )
        self.assertFalse(rejected_modify.accepted)
        self.assertNotIn(
            "replacement_reader",
            env.model_admissible_action_targets().get("modify_agent", {}).get(
                "agent_ids", []
            ),
        )
        rejected_route = await env.step(
            '{"action":"set_relation","source_id":"replacement_reader",'
            '"target_id":"reasoner","source_to_target":true,'
            '"target_to_source":false}'
        )
        self.assertFalse(rejected_route.accepted)
        rejected_delete = await env.step(
            '{"action":"delete_agent","agent_id":"replacement_reader"}'
        )
        self.assertFalse(rejected_delete.accepted)
        self.assertTrue(env.graph.has_node("replacement_reader"))

    async def test_tc1_stalled_replacement_closes_recursive_add_domain(
        self,
    ) -> None:
        graph = _trivia_semantic_graph()
        graph.add_agent(
            AgentNode(
                "failed_reader",
                "cheap",
                "retrieve grounded evidence",
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
            runtime=_trivia_semantic_runtime(registry, _ImmediateGateway()),
            graph=graph,
            problem="Who wrote the novel?",
            execute_on_edit=False,
            max_agents=8,
            semantic_protocol=QA_VERIFIED_ANSWER_LINEAGE_PROTOCOL,
            recovery_policy="preserve_diagnose_repair_augment",
            required_evidence_tool_id=QA_RETRIEVAL_TOOL_ID,
        )
        env._failed_agent_ids.add("failed_reader")
        env._repair_exhausted_agent_ids.add("failed_reader")
        env._latest_failure_record_by_agent["failed_reader"] = (
            _test_retrieval_failure_record(
                graph,
                agent_id="failed_reader",
                strategies=(
                    "initial_retrieval",
                    "spelling_normalization",
                ),
                passage_ids=("same-public-read",),
            )
        )
        added = await env.step(
            json.dumps(
                {
                    "action": "add_subgraph",
                    "agents": [
                        {
                            "agent_id": "replacement_reader",
                            "model_id": "cheap",
                            "contract": (
                                _QA_EVIDENCE_RETRIEVER_RECOVERY_CONTRACT
                            ),
                            "completion_condition": (
                                _QA_EVIDENCE_RETRIEVER_RECOVERY_COMPLETION
                            ),
                            "role_family": "evidence_retriever",
                            "allowed_tools": [QA_RETRIEVAL_TOOL_ID],
                            "execution_mode": "react",
                            "artifact_type": "retrieval_evidence",
                        }
                    ],
                    "relations": [],
                    "output_agent_id": None,
                }
            )
        )
        self.assertTrue(added.accepted, added.feedback)
        env._failed_agent_ids.add("replacement_reader")
        env._repair_exhausted_agent_ids.add("replacement_reader")
        env._latest_failure_record_by_agent["replacement_reader"] = (
                _test_retrieval_failure_record(
                    env.graph,
                    agent_id="replacement_reader",
                    strategies=(
                        "initial_retrieval",
                        "spelling_normalization",
                    ),
                    passage_ids=("same-public-read",),
                )
        )

        self.assertEqual(
            {},
            env._repair_exhausted_auxiliary_replacement_domains(),
        )
        self.assertNotIn(
            "add_subgraph",
            env.model_admissible_action_types(),
        )

    def test_tc1_recovery_saturated_retriever_closes_noop_modify_domain(
        self,
    ) -> None:
        graph = AgentGraph(
            [
                AgentNode(
                    "failed_reader",
                    "cheap",
                    "retrieve grounded evidence",
                    role_family="evidence_retriever",
                    allowed_tools=(QA_RETRIEVAL_TOOL_ID,),
                    execution_mode="react",
                    artifact_type="retrieval_evidence",
                ),
                AgentNode(
                    "replacement_reader",
                    "cheap",
                    _QA_EVIDENCE_RETRIEVER_RECOVERY_CONTRACT,
                    role_family="evidence_retriever",
                    allowed_tools=(QA_RETRIEVAL_TOOL_ID,),
                    execution_mode="react",
                    artifact_type="retrieval_evidence",
                    completion_condition=(
                        _QA_EVIDENCE_RETRIEVER_RECOVERY_COMPLETION
                    ),
                ),
            ]
        )
        registry = make_registry()
        env = AgentWorkflowEnv(
            registry,
            runtime=_trivia_semantic_runtime(registry, _ImmediateGateway()),
            graph=graph,
            problem="Who wrote the novel?",
            execute_on_edit=False,
            max_agents=8,
            semantic_protocol=QA_VERIFIED_ANSWER_LINEAGE_PROTOCOL,
            recovery_policy="preserve_diagnose_repair_augment",
            required_evidence_tool_id=QA_RETRIEVAL_TOOL_ID,
        )
        env._failed_agent_ids.add("failed_reader")
        env._repair_exhausted_agent_ids.add("failed_reader")
        env._latest_failure_record_by_agent["failed_reader"] = (
            _test_retrieval_failure_record(
                graph,
                agent_id="failed_reader",
                passage_ids=("same-public-read",),
            )
        )
        replacement_failure = _test_retrieval_failure_record(
            graph,
            agent_id="replacement_reader",
            passage_ids=("same-public-read",),
        )

        env._record_failure_state(
            (replacement_failure,),
            current_agent_ids={"failed_reader", "replacement_reader"},
        )

        self.assertIn(
            "replacement_reader",
            env._repair_exhausted_agent_ids,
        )
        self.assertEqual(
            {},
            env._triviaqa_evidence_retriever_recovery_field_values(
                "replacement_reader"
            ),
        )
        self.assertEqual(
            {},
            env._repair_exhausted_auxiliary_replacement_domains(),
        )
        self.assertEqual((), env.model_admissible_action_types())
        self.assertEqual({}, env.model_admissible_action_targets())

        # The live MODIFY projection is independently fail-closed even if an
        # old receipt omitted the saturated repair-exhausted marker.
        env._repair_exhausted_agent_ids.discard("replacement_reader")
        self.assertEqual((), env._model_admissible_modify_agent_ids())

    def test_triviaqa_valid_retriever_allows_missing_role_after_repair_failure(
        self,
    ) -> None:
        complete = _trivia_semantic_graph()
        graph = AgentGraph(
            [node for node in complete.nodes if node.id != "formatter"],
            [
                relation
                for relation in complete.relations
                if "formatter"
                not in {relation.source_id, relation.target_id}
            ],
        )
        graph.add_agent(
            AgentNode(
                "failed_repair",
                "cheap",
                "repair an incomplete evidence retrieval",
                role_family="repair",
                allowed_tools=(QA_RETRIEVAL_TOOL_ID,),
                execution_mode="react",
                artifact_type="repair_evidence",
            )
        )
        registry = make_registry()
        env = AgentWorkflowEnv(
            registry,
            runtime=_trivia_semantic_runtime(registry, _ImmediateGateway()),
            graph=graph,
            problem="What is the capital of France?",
            execute_on_edit=False,
            max_agents=8,
            semantic_protocol=QA_VERIFIED_ANSWER_LINEAGE_PROTOCOL,
            recovery_policy="preserve_diagnose_repair_augment",
            required_evidence_tool_id=QA_RETRIEVAL_TOOL_ID,
            require_format_agent=True,
        )
        env._progressive_outputs["reader"] = (
            _test_evidence_retriever_artifact("valid-public-read")
        )
        env._progressive_output_metadata["reader"] = {
            "tool_receipts": [_test_read_receipt("valid-public-read")],
        }
        env._failed_agent_ids.add("failed_repair")
        env._repair_exhausted_agent_ids.add("failed_repair")
        env._latest_failure_record_by_agent["failed_repair"] = (
            _test_retrieval_failure_record(
                graph,
                agent_id="failed_repair",
                public_error_code="knowledge_base_coverage_failure",
                bounded_schedule_exhausted=True,
            )
        )

        self.assertTrue(env._has_valid_evidence_retriever_artifact())
        self.assertEqual(("add_subgraph",), env.model_admissible_action_types())
        add_domain = env.model_admissible_action_targets()["add_subgraph"]
        self.assertEqual(
            ["format"],
            add_domain["admitted_new_role_families"],
        )

    def test_tc5_kbc_at_eight_of_eight_is_typed_terminal_not_add(self) -> None:
        graph = _trivia_semantic_graph()
        failed_reader_ids = tuple(f"failed_reader_{index}" for index in range(4))
        for agent_id in failed_reader_ids:
            graph.add_agent(
                AgentNode(
                    agent_id,
                    "cheap",
                    "retrieve grounded evidence",
                    role_family="evidence_retriever",
                    allowed_tools=(QA_RETRIEVAL_TOOL_ID,),
                    execution_mode="react",
                    artifact_type="retrieval_evidence",
                )
            )
            graph.set_relation(agent_id, "reasoner", True, False)
        self.assertEqual(8, len(graph.nodes))
        registry = make_registry()
        env = AgentWorkflowEnv(
            registry,
            runtime=_trivia_semantic_runtime(registry, _ImmediateGateway()),
            graph=graph,
            problem="Who wrote the novel?",
            execute_on_edit=False,
            max_agents=8,
            semantic_protocol=QA_VERIFIED_ANSWER_LINEAGE_PROTOCOL,
            recovery_policy="preserve_diagnose_repair_augment",
            required_evidence_tool_id=QA_RETRIEVAL_TOOL_ID,
        )
        env._failed_agent_ids.update((*failed_reader_ids, "reasoner"))
        env._repair_exhausted_agent_ids.update(
            (*failed_reader_ids, "reasoner")
        )
        for index, agent_id in enumerate(failed_reader_ids):
            env._latest_failure_record_by_agent[agent_id] = (
                _test_retrieval_failure_record(
                    graph,
                    agent_id=agent_id,
                    public_error_code="knowledge_base_coverage_failure",
                    strategies=(
                        "initial_retrieval",
                        "spelling_normalization",
                        "alias_expansion",
                        "entity_disambiguation",
                        "relation_query_rewriting",
                    ),
                    passage_ids=(f"coverage-read-{index}",),
                    bounded_schedule_exhausted=True,
                )
            )

        self.assertEqual(
            {},
            env._repair_exhausted_auxiliary_replacement_domains(),
        )
        self.assertEqual((), env.model_admissible_action_types())
        self.assertNotIn(
            "add_subgraph",
            env.model_admissible_action_targets(),
        )

    async def test_tc5_kbc_below_capacity_is_typed_terminal_not_relation_cycle(
        self,
    ) -> None:
        graph = _trivia_semantic_graph()
        registry = make_registry()
        env = AgentWorkflowEnv(
            registry,
            runtime=_trivia_semantic_runtime(registry, _ImmediateGateway()),
            graph=graph,
            problem="Who wrote the novel?",
            execute_on_edit=False,
            max_agents=8,
            semantic_protocol=QA_VERIFIED_ANSWER_LINEAGE_PROTOCOL,
            recovery_policy="preserve_diagnose_repair_augment",
            required_evidence_tool_id=QA_RETRIEVAL_TOOL_ID,
        )
        record = _test_retrieval_failure_record(
            graph,
            agent_id="reader",
            public_error_code="knowledge_base_coverage_failure",
            strategies=(
                "initial_retrieval",
                "spelling_normalization",
                "alias_expansion",
                "entity_disambiguation",
                "relation_query_rewriting",
            ),
            passage_ids=("retained-public-read",),
            bounded_schedule_exhausted=True,
        )
        env._failed_agent_ids.add("reader")
        env._repair_exhausted_agent_ids.add("reader")
        env._latest_failure_record_by_agent["reader"] = record

        self.assertEqual(
            "knowledge_base_coverage_failure",
            env._typed_retrieval_failure_category(record),
        )
        self.assertTrue(env._agent_has_successful_read_receipt("reader"))
        self.assertEqual(
            {},
            env._repair_exhausted_auxiliary_replacement_domains(),
        )
        self.assertEqual([], env._model_admissible_relation_candidates())
        self.assertEqual((), env.model_admissible_action_types())
        self.assertEqual({}, env.model_admissible_action_targets())
        self.assertEqual([], env.recovery_state()["preferred_actions"])

        revision = env.revision
        rejected = await env.step(
            '{"action":"set_relation","source_id":"reader",'
            '"target_id":"reasoner","source_to_target":false,'
            '"target_to_source":false}'
        )
        self.assertFalse(rejected.accepted)
        self.assertIn("no admissible semantic recovery relation", rejected.feedback)
        self.assertEqual(revision, env.revision)
        self.assertIn("reader", env.graph.directed_predecessors("reasoner"))

    async def test_failed_artifact_free_retriever_cannot_repair_reachability(
        self,
    ) -> None:
        graph = _trivia_semantic_graph()
        registry = make_registry()
        env = AgentWorkflowEnv(
            registry,
            runtime=_trivia_semantic_runtime(registry, _ImmediateGateway()),
            graph=graph,
            problem="Who wrote the novel?",
            execute_on_edit=False,
            semantic_protocol=QA_VERIFIED_ANSWER_LINEAGE_PROTOCOL,
            recovery_policy="preserve_diagnose_repair_augment",
            required_evidence_tool_id=QA_RETRIEVAL_TOOL_ID,
        )
        env._failed_agent_ids.add("reasoner")
        env._repair_exhausted_agent_ids.add("reasoner")
        added = await env.step(
            json.dumps(
                {
                    "action": "add_subgraph",
                    "agents": [
                        {
                            "agent_id": "failed_reader",
                            "model_id": "cheap",
                            "contract": (
                                _QA_EVIDENCE_RETRIEVER_RECOVERY_CONTRACT
                            ),
                            "completion_condition": (
                                _QA_EVIDENCE_RETRIEVER_RECOVERY_COMPLETION
                            ),
                            "role_family": "evidence_retriever",
                            "allowed_tools": [QA_RETRIEVAL_TOOL_ID],
                            "execution_mode": "react",
                            "artifact_type": "retrieval_evidence",
                        }
                    ],
                    "relations": [],
                    "output_agent_id": None,
                }
            )
        )
        self.assertTrue(added.accepted, added.feedback)
        env._failed_agent_ids.add("failed_reader")
        env._repair_exhausted_agent_ids.add("failed_reader")
        env._latest_failure_record_by_agent["failed_reader"] = (
            _test_retrieval_failure_record(
                env.graph,
                agent_id="failed_reader",
                public_error_code="knowledge_base_coverage_failure",
                bounded_schedule_exhausted=True,
            )
        )

        self.assertEqual(
            {"failed_reader"},
            set(env._terminal_unreachable_agent_ids()),
        )
        self.assertEqual(
            [],
            env._terminal_reachability_relation_candidates(),
        )
        rejected = await env.step(
            '{"action":"set_relation","source_id":"failed_reader",'
            '"target_id":"reasoner","source_to_target":true,'
            '"target_to_source":false}'
        )
        self.assertFalse(rejected.accepted)
        self.assertIn("artifact-free auxiliary", rejected.feedback)

    async def test_replacement_routes_only_to_reasoner_then_preserves_takeover(
        self,
    ) -> None:
        graph = _hotpot_semantic_graph()
        for agent_id in ("failed_reader", "replacement_reader"):
            graph.add_agent(
                AgentNode(
                    agent_id,
                    "cheap",
                    "retrieve grounded evidence",
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
        replacement_receipt = _test_read_receipt("replacement-public")
        replacement_artifact = _test_evidence_retriever_artifact(
            "replacement-public"
        )
        env._progressive_outputs["replacement_reader"] = replacement_artifact
        env._progressive_output_metadata["replacement_reader"] = {
            "tool_receipts": [replacement_receipt]
        }
        env._failed_agent_ids.update({"failed_reader", "reasoner"})
        env._repair_exhausted_agent_ids.update(
            {"failed_reader", "reasoner"}
        )
        env._latest_failure_record_by_agent["failed_reader"] = (
            _test_retrieval_failure_record(
                graph,
                agent_id="failed_reader",
                strategies=("initial_retrieval",),
                passage_ids=("failed-public",),
            )
        )

        expected_route = {
            "source_id": "replacement_reader",
            "target_id": "reasoner",
            "source_to_target": True,
            "target_to_source": False,
        }
        self.assertEqual(
            [expected_route],
            env._repair_exhausted_relation_candidates(),
        )
        all_candidates = env._all_model_admissible_relation_candidates()
        failed_target = {
            "source_id": "replacement_reader",
            "target_id": "failed_reader",
            "source_to_target": True,
            "target_to_source": False,
        }
        self.assertTrue(
            env._relation_routes_replacement_outside_reasoner(failed_target)
        )
        self.assertNotIn(
            failed_target,
            env._model_admissible_relation_candidates(),
        )

        routed = await env.step(
            json.dumps({"action": "set_relation", **expected_route})
        )
        self.assertTrue(routed.accepted, routed.feedback)
        self.assertEqual(("delete_agent",), env.model_admissible_action_types())
        deleted = await env.step(
            '{"action":"delete_agent","agent_id":"failed_reader"}'
        )
        self.assertTrue(deleted.accepted, deleted.feedback)
        self.assertEqual(
            replacement_artifact,
            env._progressive_outputs["replacement_reader"],
        )
        self.assertEqual(
            [replacement_receipt],
            env._progressive_output_metadata["replacement_reader"][
                "tool_receipts"
            ],
        )

    async def test_auxiliary_replacement_add_is_one_isolated_domain_agent(
        self,
    ) -> None:
        graph = _trivia_semantic_graph()
        graph.add_agent(
            AgentNode(
                "failed_reader",
                "cheap",
                "retrieve evidence for the target entity and relation",
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
            runtime=_trivia_semantic_runtime(registry, _ImmediateGateway()),
            graph=graph,
            problem="Who wrote the novel?",
            execute_on_edit=False,
            max_agents=8,
            semantic_protocol=QA_VERIFIED_ANSWER_LINEAGE_PROTOCOL,
            recovery_policy="preserve_diagnose_repair_augment",
            required_evidence_tool_id=QA_RETRIEVAL_TOOL_ID,
        )
        env._failed_agent_ids.add("failed_reader")
        env._repair_exhausted_agent_ids.add("failed_reader")
        env._latest_failure_record_by_agent["failed_reader"] = (
            _test_retrieval_failure_record(
                graph,
                agent_id="failed_reader",
                public_error_code="retrieval_recall_failure",
                strategies=("initial_retrieval",),
            )
        )

        self.assertIn("add_subgraph", env.model_admissible_action_types())
        add_domain = env.model_admissible_action_targets()["add_subgraph"]
        self.assertEqual(1, add_domain["max_new_agents"])
        self.assertEqual([], add_domain["relations"])
        self.assertIsNone(add_domain["output_agent_id"])
        retriever_domain = add_domain["role_constraints"][
            "evidence_retriever"
        ]
        self.assertEqual(
            [_QA_EVIDENCE_RETRIEVER_RECOVERY_CONTRACT],
            retriever_domain["contracts"],
        )
        self.assertEqual(
            [_QA_EVIDENCE_RETRIEVER_RECOVERY_COMPLETION],
            retriever_domain["completion_conditions"],
        )

        replacement_spec = {
            "model_id": "cheap",
            "contract": _QA_EVIDENCE_RETRIEVER_RECOVERY_CONTRACT,
            "completion_condition": (
                _QA_EVIDENCE_RETRIEVER_RECOVERY_COMPLETION
            ),
            "role_family": "evidence_retriever",
            "allowed_tools": [QA_RETRIEVAL_TOOL_ID],
            "execution_mode": "react",
            "artifact_type": "retrieval_evidence",
        }
        rejected_multiple = await env.step(
            json.dumps(
                {
                    "action": "add_subgraph",
                    "agents": [
                        {"agent_id": "replacement_a", **replacement_spec},
                        {"agent_id": "replacement_b", **replacement_spec},
                    ],
                    "relations": [],
                }
            )
        )
        self.assertFalse(rejected_multiple.accepted)
        self.assertIn("exactly one", rejected_multiple.feedback)
        self.assertFalse(env.graph.has_node("replacement_a"))
        self.assertFalse(env.graph.has_node("replacement_b"))

        accepted = await env.step(
            json.dumps(
                {
                    "action": "add_subgraph",
                    "agents": [
                        {"agent_id": "replacement_reader", **replacement_spec}
                    ],
                    "relations": [],
                    "output_agent_id": None,
                }
            )
        )
        self.assertTrue(accepted.accepted, accepted.feedback)
        self.assertTrue(env.graph.has_node("replacement_reader"))
        self.assertEqual(
            (),
            env.graph.directed_predecessors("replacement_reader"),
        )
        self.assertEqual(
            (),
            env._directed_successors(env.graph, "replacement_reader"),
        )
        env._failed_agent_ids.add("replacement_reader")
        env._repair_exhausted_agent_ids.add("replacement_reader")
        env._latest_failure_record_by_agent["replacement_reader"] = (
            _test_retrieval_failure_record(
                env.graph,
                agent_id="replacement_reader",
                public_error_code="retrieval_recall_failure",
                strategies=("initial_retrieval",),
                bounded_schedule_exhausted=True,
            )
        )
        self.assertEqual(
            {},
            env._repair_exhausted_auxiliary_replacement_domains(),
        )
        self.assertNotIn(
            "add_subgraph",
            env.model_admissible_action_types(),
        )

    async def test_successful_isolated_replacement_routes_before_missing_role_add(
        self,
    ) -> None:
        """A completed replacement closes its ADD domain and routes next."""

        graph = AgentGraph(
            [
                AgentNode(
                    "failed_reader",
                    "cheap",
                    "retrieve grounded evidence",
                    role_family="evidence_retriever",
                    allowed_tools=(QA_RETRIEVAL_TOOL_ID,),
                    execution_mode="react",
                    artifact_type="retrieval_evidence",
                ),
                AgentNode(
                    "reasoner",
                    "balanced",
                    "bind grounded evidence to the requested answer slot",
                    role_family="reasoner",
                    allowed_tools=(),
                    execution_mode="reasoning",
                    artifact_type="semantic_candidate",
                ),
                AgentNode(
                    "verifier",
                    "balanced",
                    "verify evidence and semantic answer lineage",
                    role_family="verifier",
                    allowed_tools=(),
                    execution_mode="reasoning",
                    artifact_type="verified_semantic_answer",
                ),
                AgentNode(
                    "replacement_reader",
                    "cheap",
                    "retrieve replacement grounded evidence",
                    role_family="evidence_retriever",
                    allowed_tools=(QA_RETRIEVAL_TOOL_ID,),
                    execution_mode="react",
                    artifact_type="retrieval_evidence",
                ),
            ],
            [AgentRelation("failed_reader", "reasoner", True, False)],
        )
        registry = make_registry()
        env = AgentWorkflowEnv(
            registry,
            runtime=_trivia_semantic_runtime(registry, _ImmediateGateway()),
            graph=graph,
            problem="What is the capital of France?",
            execute_on_edit=False,
            max_agents=8,
            semantic_protocol=QA_VERIFIED_ANSWER_LINEAGE_PROTOCOL,
            recovery_policy="preserve_diagnose_repair_augment",
            required_evidence_tool_id=QA_RETRIEVAL_TOOL_ID,
            require_format_agent=True,
        )
        env._failed_agent_ids.add("failed_reader")
        env._repair_exhausted_agent_ids.add("failed_reader")
        env._latest_failure_record_by_agent["failed_reader"] = (
            _test_retrieval_failure_record(
                graph,
                agent_id="failed_reader",
                strategies=("initial_retrieval",),
                passage_ids=("failed-public",),
            )
        )
        env._progressive_outputs["replacement_reader"] = (
            _test_evidence_retriever_artifact("replacement-public")
        )
        env._progressive_output_metadata["replacement_reader"] = {
            "tool_receipts": [_test_read_receipt("replacement-public")]
        }

        expected_route = {
            "source_id": "replacement_reader",
            "target_id": "reasoner",
            "source_to_target": True,
            "target_to_source": False,
        }
        self.assertEqual(
            {},
            env._repair_exhausted_auxiliary_replacement_domains(),
        )
        self.assertEqual(
            [expected_route],
            env._required_evidence_ingress_relation_candidates(),
        )
        self.assertEqual(
            ("set_relation",),
            env.model_admissible_action_types(),
        )
        self.assertEqual(
            [expected_route],
            env.model_admissible_action_targets()["set_relation"][
                "candidates"
            ],
        )

        routed = await env.step(
            json.dumps({"action": "set_relation", **expected_route})
        )
        self.assertTrue(routed.accepted, routed.feedback)
        self.assertEqual(
            ("add_subgraph",),
            env.model_admissible_action_types(),
        )
        self.assertEqual(
            ["format"],
            env.model_admissible_action_targets()["add_subgraph"][
                "admitted_new_role_families"
            ],
        )

    def test_multigeneration_replacement_handoff_uses_most_advanced_public_state(
        self,
    ) -> None:
        graph = _trivia_semantic_graph()
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
            runtime=_trivia_semantic_runtime(registry, _ImmediateGateway()),
            graph=graph,
            problem="Who wrote the novel?",
            execute_on_edit=False,
            max_agents=8,
            semantic_protocol=QA_VERIFIED_ANSWER_LINEAGE_PROTOCOL,
            recovery_policy="preserve_diagnose_repair_augment",
            required_evidence_tool_id=QA_RETRIEVAL_TOOL_ID,
        )
        first_receipt = _test_read_receipt("first-public")
        newest_receipts = [
            _test_read_receipt("newest-public-1"),
            _test_read_receipt("newest-public-2"),
        ]
        env._failed_agent_ids.update({"failed_reader", "replacement_reader"})
        env._repair_exhausted_agent_ids.update(
            {"failed_reader", "replacement_reader"}
        )
        env._failure_continuations.update(
            {
                "failed_reader": {
                    "execution_phase": "single",
                    "tool_receipts": [first_receipt],
                    "private_reasoning": "must not cross the Agent boundary",
                },
                "replacement_reader": {
                    "execution_phase": "single",
                    "tool_receipts": newest_receipts,
                    "private_reasoning": "must not cross the Agent boundary",
                },
            }
        )
        action = env.parser.parse(
            json.dumps(
                {
                    "action": "add_subgraph",
                    "agents": [
                        {
                            "agent_id": "next_reader",
                            "model_id": "cheap",
                            "contract": "continue grounded public retrieval",
                            "role_family": "evidence_retriever",
                            "allowed_tools": [QA_RETRIEVAL_TOOL_ID],
                            "execution_mode": "react",
                            "artifact_type": "retrieval_evidence",
                        }
                    ],
                    "relations": [],
                }
            )
        )

        handoff = env._recovery_continuation_handoff(action)

        self.assertEqual(["next_reader"], list(handoff))
        projected = handoff["next_reader"]
        self.assertEqual(
            "replacement_reader",
            projected["continuation_source_agent_id"],
        )
        self.assertEqual(newest_receipts, projected["tool_receipts"])
        self.assertNotIn("private_reasoning", projected)

    def test_exhausted_tool_plan_is_not_handed_to_replacement_agent(
        self,
    ) -> None:
        graph = _trivia_semantic_graph()
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
            runtime=_trivia_semantic_runtime(registry, _ImmediateGateway()),
            graph=graph,
            problem="Who wrote the novel?",
            execute_on_edit=False,
            max_agents=8,
            semantic_protocol=QA_VERIFIED_ANSWER_LINEAGE_PROTOCOL,
            recovery_policy="preserve_diagnose_repair_augment",
            required_evidence_tool_id=QA_RETRIEVAL_TOOL_ID,
        )
        receipt = _test_read_receipt("exhausted-public")
        failure = AgentFailureRecord(
            request_id="exhausted-reader",
            agent_id="failed_reader",
            phase=ExecutionPhase.SINGLE,
            graph_revision=graph.revision,
            error_type="ReactExecutionError",
            message="bounded Tool plan exhausted",
            metadata={
                "tool_receipts": [receipt],
                "tool_plan_exhausted": True,
            },
        )
        continuation = env._failure_continuation_candidate(failure)
        self.assertIsNotNone(continuation)
        assert continuation is not None
        self.assertIs(True, continuation["tool_plan_exhausted"])
        env._failed_agent_ids.add("failed_reader")
        env._repair_exhausted_agent_ids.add("failed_reader")
        env._failure_continuations["failed_reader"] = continuation
        action = env.parser.parse(
            json.dumps(
                {
                    "action": "add_subgraph",
                    "agents": [
                        {
                            "agent_id": "next_reader",
                            "model_id": "cheap",
                            "contract": "start a fresh bounded retrieval plan",
                            "role_family": "evidence_retriever",
                            "allowed_tools": [QA_RETRIEVAL_TOOL_ID],
                            "execution_mode": "react",
                            "artifact_type": "retrieval_evidence",
                        }
                    ],
                    "relations": [],
                }
            )
        )

        self.assertEqual({}, env._recovery_continuation_handoff(action))

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
                "terminal_failure_diagnosis": {
                    "public_error_code": "retrieval_strategy_failure",
                    "bounded_schedule_exhausted": False,
                    "retrieval_strategy_schedule_prefix": [
                        "initial_retrieval"
                    ],
                },
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
        observed_execution_graphs: list[
            tuple[tuple[str, ...], str | None, int]
        ] = []
        observed_execution_kwargs: list[dict[str, object]] = []

        async def replacement_failure(
            candidate_graph: AgentGraph,
            *args: object,
            **kwargs: object,
        ) -> AgentRuntimeResult:
            del args
            observed_execution_graphs.append(
                (
                    tuple(node.id for node in candidate_graph.nodes),
                    candidate_graph.output_agent_id,
                    candidate_graph.revision,
                )
            )
            observed_execution_kwargs.append(dict(kwargs))
            raise AgentRuntimeError(
                "replacement Retriever exhausted its bounded ReAct execution",
                failure_records=(
                    AgentFailureRecord(
                        request_id="tc10-replacement-exhausted",
                        agent_id="replacement_reader",
                        phase=ExecutionPhase.SINGLE,
                        graph_revision=candidate_graph.revision,
                        error_type="ReactExecutionError",
                        message=(
                            "react agent 'replacement_reader' exhausted 8 turns"
                        ),
                        metadata={
                            "continuation_source_agent_id": "failed_reader",
                            "react_trace": [tool_trace],
                            "tool_receipts": [receipt],
                        },
                    ),
                ),
                pending_agent_ids=("replacement_reader",),
            )

        async def replacement_success(
            candidate_graph: AgentGraph,
            *args: object,
            **kwargs: object,
        ) -> AgentRuntimeResult:
            del args
            observed_execution_graphs.append(
                (
                    tuple(node.id for node in candidate_graph.nodes),
                    candidate_graph.output_agent_id,
                    candidate_graph.revision,
                )
            )
            observed_execution_kwargs.append(dict(kwargs))
            observed_failure_metadata.update(kwargs["prior_failure_metadata"])
            return AgentRuntimeResult(
                run_id="tc10-replacement",
                graph_revision=candidate_graph.revision,
                output_agent_id=candidate_graph.output_agent_id,
                final_answer=None,
                outputs={
                    "replacement_reader": _test_evidence_retriever_artifact(
                        "tc10-public"
                    )
                },
                output_metadata={
                    "replacement_reader": {"tool_receipts": [receipt]}
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
        with patch.object(runtime, "execute", side_effect=replacement_failure):
            failed_add = await env.step(action_payload)
        self.assertTrue(failed_add.accepted, failed_add.feedback)
        self.assertEqual(1, len(failed_add.execution_failure_records))
        self.assertIn("failed_reader", env._failed_agent_ids)
        self.assertIn("failed_reader", env._repair_exhausted_agent_ids)
        self.assertEqual(
            "tc10-reader-exhausted",
            env._latest_failure_record_by_agent["failed_reader"].request_id,
        )
        self.assertIn("replacement_reader", env._unresolved_dirty_agents)
        self.assertEqual(("modify_agent",), env.model_admissible_action_types())

        with patch.object(runtime, "execute", side_effect=replacement_success):
            result = await env.step(
                json.dumps(
                    {
                        "action": "modify_agent",
                        "agent_id": "replacement_reader",
                        "contract": (
                            "continue public grounded evidence retrieval with "
                            "the preserved Tool receipts"
                        ),
                    }
                )
            )

        self.assertTrue(result.accepted, result.feedback)
        self.assertEqual(("replacement_reader",), observed_execution_graphs[0][0])
        self.assertIsNone(observed_execution_graphs[0][1])
        self.assertEqual(
            {"replacement_reader"},
            observed_execution_kwargs[0]["dirty_agents"],
        )
        self.assertEqual({}, observed_execution_kwargs[0]["prior_outputs"])
        self.assertFalse(observed_execution_kwargs[0]["format_output_agent"])
        self.assertEqual(("replacement_reader",), observed_execution_graphs[1][0])
        self.assertIsNone(observed_execution_graphs[1][1])
        self.assertEqual(env.graph.revision, observed_execution_graphs[1][2])
        self.assertEqual(
            {"replacement_reader"},
            observed_execution_kwargs[1]["dirty_agents"],
        )
        self.assertEqual({}, observed_execution_kwargs[1]["prior_outputs"])
        self.assertFalse(observed_execution_kwargs[1]["format_output_agent"])
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
            {node.id for node in env.graph.nodes},
            set(observed_execution_graphs[2][0]),
        )
        self.assertEqual("formatter", observed_execution_graphs[2][1])
        self.assertTrue(observed_execution_kwargs[2]["format_output_agent"])
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

    def test_reasoner_scope_relation_and_entity_gate_reuses_read_receipts(
        self,
    ) -> None:
        question = (
            "In which decade did Billboard magazine first publish an American "
            "hit chart?"
        )

        def artifact(evidence_span: str, relation: str) -> str:
            return json.dumps(
                {
                    "question_scope": question,
                    "answer_slot": {
                        "answer_type": "date",
                        "answer_cardinality": "single",
                        "qualifiers": ["first"],
                        "proposition_index": 0,
                        "answer_field": "object_or_attribute_value",
                    },
                    "evidence_propositions": [
                        {
                            "subject": "Billboard magazine",
                            "relation": relation,
                            "object_or_attribute_value": "1961",
                            "qualifiers": ["first"],
                            "evidence_span": evidence_span,
                        }
                    ],
                    "multi_hop_chain": [
                        "identify the requested publication event",
                        "map 1961 to the 1960s",
                    ],
                    "candidate_answer": "1960s",
                    "evidence": [evidence_span],
                }
            )

        read_text = (
            "Billboard magazine published an American hit chart in 1961."
        )
        missing_scope = AgentWorkflowEnv._reasoner_evidence_provenance_issue(
            artifact(read_text, "published"),
            [read_text],
            require_answer_binding=True,
            original_question=question,
        )
        self.assertIn("semantic scope", missing_scope or "")

        grounded_span = (
            "Billboard magazine first published an American hit chart in 1961."
        )
        wrong_relation = AgentWorkflowEnv._reasoner_evidence_provenance_issue(
            artifact(grounded_span, "launched"),
            [grounded_span],
            require_answer_binding=True,
            original_question=question,
        )
        self.assertIn("relation is not grounded", wrong_relation or "")
        self.assertIsNone(
            AgentWorkflowEnv._reasoner_evidence_provenance_issue(
                artifact(grounded_span, "first published"),
                [grounded_span],
                require_answer_binding=True,
                original_question=question,
            )
        )

    def test_reasoner_accepts_receipt_grounded_compound_relation_realization(
        self,
    ) -> None:
        question = (
            "Which American-born Sinclair won the Nobel Prize for Literature "
            "in 1930?"
        )
        evidence_span = (
            "Harry Sinclair Lewis was an American novelist. In 1930, he became "
            "the first writer from the United States to receive the Nobel Prize "
            "in Literature."
        )

        def artifact(
            relation: str,
            object_or_attribute_value: str = (
                "the first writer from the United States to receive the Nobel "
                "Prize in Literature"
            ),
        ) -> str:
            return json.dumps(
                {
                    "question_scope": question,
                    "answer_slot": {
                        "answer_type": "entity",
                        "answer_cardinality": "single",
                        "qualifiers": [],
                        "proposition_index": 0,
                        "answer_field": "subject",
                    },
                    "evidence_propositions": [
                        {
                            "subject": "Harry Sinclair Lewis",
                            "relation": relation,
                            "object_or_attribute_value": object_or_attribute_value,
                            "qualifiers": ["in 1930"],
                            "evidence_span": evidence_span,
                        }
                    ],
                    "multi_hop_chain": [
                        "bind Sinclair to Harry Sinclair Lewis",
                        "bind the requested prize relation",
                    ],
                    "candidate_answer": "Harry Sinclair Lewis",
                    "evidence": [evidence_span],
                }
            )

        self.assertIsNone(
            AgentWorkflowEnv._reasoner_evidence_provenance_issue(
                artifact("became"),
                [evidence_span],
                require_answer_binding=True,
                original_question=question,
            )
        )
        unrelated = AgentWorkflowEnv._reasoner_evidence_provenance_issue(
            artifact("was", "an American novelist"),
            [evidence_span],
            require_answer_binding=True,
            original_question=question,
        )
        self.assertIn("do not preserve the requested relation", unrelated or "")

    def test_reasoner_accepts_relation_split_across_relevant_qamemory_pair(
        self,
    ) -> None:
        question = (
            "Which innovation for the car was developed by Prince Henry of "
            "Prussia in 1911?"
        )
        evidence_span = (
            "The innovation developed by Prince Henry of Prussia in 1911 is "
            "Windshield wipers."
        )
        memory_id = "qa-memory-car-innovation"
        receipt = {
            "tool_id": "triviaqa.qa_memory",
            "tool_version": "test-v1",
            "request": {
                "action": "read",
                "arguments": {"memory_id": memory_id},
            },
            "result": {
                "value": {
                    "operation": "read",
                    "memory_id": memory_id,
                    "memory": {
                        "memory_id": memory_id,
                        "title": (
                            "What car innovation was created by Prince Henry "
                            "of Prussia in 1911?"
                        ),
                        "text": (
                            "Question: What car innovation was created by "
                            "Prince Henry of Prussia in 1911?\nAnswer: "
                            + evidence_span
                        ),
                        "paraphrase_question": (
                            "What car innovation was created by Prince Henry "
                            "of Prussia in 1911?"
                        ),
                        "paraphrase_answer_statement": evidence_span,
                    },
                },
                "completed": True,
            },
            "error_type": None,
        }
        read_text = AgentWorkflowEnv._successful_read_text(
            receipt,
            "triviaqa.qa_memory",
        )
        self.assertIsNotNone(read_text)
        assert read_text is not None
        artifact = json.dumps(
            {
                "question_scope": question,
                "answer_slot": {
                    "answer_type": "entity",
                    "answer_cardinality": "single",
                    "qualifiers": [
                        "developed by Prince Henry of Prussia",
                        "in 1911",
                    ],
                    "proposition_index": 0,
                    "answer_field": "object_or_attribute_value",
                },
                "evidence_propositions": [
                    {
                        "subject": (
                            "The innovation developed by Prince Henry of "
                            "Prussia in 1911"
                        ),
                        "relation": "is",
                        "object_or_attribute_value": "Windshield wipers",
                        "qualifiers": [],
                        "evidence_span": evidence_span,
                    }
                ],
                "multi_hop_chain": [
                    "Prince Henry of Prussia developed the innovation in 1911"
                ],
                "candidate_answer": "Windshield wipers",
                "evidence": [evidence_span],
            }
        )

        self.assertIsNone(
            AgentWorkflowEnv._reasoner_evidence_provenance_issue(
                artifact,
                [read_text],
                require_answer_binding=True,
                original_question=question,
                qa_memory_relevant_memory_ids=[memory_id],
            )
        )
        unrelated = AgentWorkflowEnv._reasoner_evidence_provenance_issue(
            artifact,
            [read_text],
            require_answer_binding=True,
            original_question=question,
            qa_memory_relevant_memory_ids=["different-memory"],
        )
        self.assertIn("do not preserve the requested relation", unrelated or "")

    def test_reasoner_accepts_exact_source_semantic_qamemory_paraphrase(
        self,
    ) -> None:
        question = "Which country is Europe's largest silk producer?"
        paraphrase_question = (
            "Which nation serves as Europe's biggest silk manufacturer?"
        )
        evidence_span = (
            "Italy is the nation that serves as Europe's biggest silk "
            "manufacturer."
        )
        memory_id = "qa-memory-silk-producer"
        source_task_id = "triviaqa:tc_silk"
        receipt = {
            "tool_id": "triviaqa.qa_memory",
            "tool_version": "test-v1",
            "request": {
                "action": "read",
                "arguments": {"memory_id": memory_id},
            },
            "result": {
                "value": {
                    "operation": "read",
                    "memory_id": memory_id,
                    "memory": {
                        "memory_id": memory_id,
                        "source_train_task_id": source_task_id,
                        "canonical_answer": "Italy",
                        "title": paraphrase_question,
                        "text": (
                            f"Question: {paraphrase_question}\n"
                            f"Answer: {evidence_span}"
                        ),
                        "paraphrase_question": paraphrase_question,
                        "paraphrase_answer_statement": evidence_span,
                        "paraphrase_provenance": {
                            "canonical_span_preserved": True,
                            "paraphrase_method": (
                                "semantic-preserving-question-and-answer-"
                                "paraphrase"
                            ),
                        },
                    },
                },
                "completed": True,
            },
            "error_type": None,
        }
        read_text = AgentWorkflowEnv._successful_read_text(
            receipt,
            "triviaqa.qa_memory",
        )
        self.assertIsNotNone(read_text)
        assert read_text is not None
        artifact = json.dumps(
            {
                "question_scope": question,
                "answer_slot": {
                    "answer_type": "entity",
                    "answer_cardinality": "single",
                    "qualifiers": [],
                    "proposition_index": 0,
                    "answer_field": "subject",
                },
                "evidence_propositions": [
                    {
                        "subject": "Italy",
                        "relation": "is",
                        "object_or_attribute_value": (
                            "the nation that serves as Europe's biggest silk "
                            "manufacturer"
                        ),
                        "qualifiers": [],
                        "evidence_span": evidence_span,
                    }
                ],
                "multi_hop_chain": [evidence_span],
                "candidate_answer": "Italy",
                "evidence": [evidence_span],
            }
        )

        self.assertIsNone(
            AgentWorkflowEnv._reasoner_evidence_provenance_issue(
                artifact,
                [read_text],
                require_answer_binding=True,
                original_question=question,
                qa_memory_relevant_memory_ids=[memory_id],
                qa_memory_expected_source_task_ids=[source_task_id],
            )
        )
        wrong_source = AgentWorkflowEnv._reasoner_evidence_provenance_issue(
            artifact,
            [read_text],
            require_answer_binding=True,
            original_question=question,
            qa_memory_relevant_memory_ids=["different-memory"],
            qa_memory_expected_source_task_ids=["triviaqa:different"],
        )
        self.assertIn(
            "do not preserve the requested relation",
            wrong_source or "",
        )

    def test_reasoner_exact_source_qamemory_binds_possessive_entity(self) -> None:
        question = "Which prince is Queen Elizabeth II's youngest son?"
        evidence_span = "Edward is the youngest son of Queen Elizabeth II."
        memory_id = "qa-memory-youngest-son"
        source_task_id = "triviaqa:tc_youngest_son"
        artifact = json.dumps(
            {
                "question_scope": question,
                "answer_slot": {
                    "answer_type": "entity",
                    "answer_cardinality": "single",
                    "qualifiers": [],
                    "proposition_index": 0,
                    "answer_field": "subject",
                },
                "evidence_propositions": [
                    {
                        "subject": "Edward",
                        "relation": "is",
                        "object_or_attribute_value": (
                            "the youngest son of Queen Elizabeth II"
                        ),
                        "qualifiers": [],
                        "evidence_span": evidence_span,
                    }
                ],
                "multi_hop_chain": [evidence_span],
                "candidate_answer": "Edward",
                "evidence": [evidence_span],
            }
        )

        self.assertIsNone(
            AgentWorkflowEnv._reasoner_evidence_provenance_issue(
                artifact,
                [
                    _ReadReceiptText(
                        (
                            "Question: Which prince is Queen Elizabeth II's "
                            "youngest offspring?\nAnswer: " + evidence_span
                        ),
                        passage_title=(
                            "Which prince is Queen Elizabeth II's youngest "
                            "offspring?"
                        ),
                        tool_id="triviaqa.qa_memory",
                        record_id=memory_id,
                        paraphrase_question=(
                            "Which prince is Queen Elizabeth II's youngest "
                            "offspring?"
                        ),
                        paraphrase_answer_statement=evidence_span,
                        source_train_task_id=source_task_id,
                        canonical_answer="Edward",
                        semantic_preserving_paraphrase=True,
                    )
                ],
                require_answer_binding=True,
                original_question=question,
                qa_memory_relevant_memory_ids=[memory_id],
                qa_memory_expected_source_task_ids=[source_task_id],
            )
        )

    def test_exact_source_qamemory_grounds_entity_across_paired_question(self) -> None:
        question = (
            "Brooks Robinson and Carl Yastrzemski hold the major league "
            "baseball record for playing the greatest number of seasons with "
            "the same team. How many years did they play-- and with what teams?"
        )
        paraphrase_question = question.replace("hold", "maintain")
        canonical_answer = (
            "23 years. Third baseman Robinson played with the Baltimore "
            "Orioles from 1955 to 1977; Carl Yastrzemski played with the "
            "Boston Red Sox from 1961 to 1983"
        )
        evidence_span = "They played for " + canonical_answer
        memory_id = "qa-memory-baseball-seasons"
        source_task_id = "triviaqa:tc_171"
        artifact = json.dumps(
            {
                "question_scope": question,
                "answer_slot": {
                    "answer_type": "number",
                    "answer_cardinality": "single",
                    "qualifiers": ["years", "teams"],
                    "proposition_index": 0,
                    "answer_field": "object_or_attribute_value",
                },
                "evidence_propositions": [
                    {
                        "subject": "Brooks Robinson",
                        "relation": "played_with",
                        "object_or_attribute_value": canonical_answer,
                        "qualifiers": ["major league baseball record"],
                        "evidence_span": evidence_span,
                    }
                ],
                "multi_hop_chain": ["bind years and both teams"],
                "candidate_answer": canonical_answer,
                "evidence": [memory_id],
            }
        )
        read_text = _ReadReceiptText(
            f"Question: {paraphrase_question}\nAnswer: {evidence_span}",
            tool_id="triviaqa.qa_memory",
            record_id=memory_id,
            paraphrase_question=paraphrase_question,
            paraphrase_answer_statement=evidence_span,
            source_train_task_id=source_task_id,
            canonical_answer=canonical_answer,
            semantic_preserving_paraphrase=True,
        )

        self.assertIsNone(
            AgentWorkflowEnv._reasoner_evidence_provenance_issue(
                artifact,
                [read_text],
                require_answer_binding=True,
                original_question=question,
                qa_memory_relevant_memory_ids=[memory_id],
                qa_memory_expected_source_task_ids=[source_task_id],
            )
        )
        wrong_source = AgentWorkflowEnv._reasoner_evidence_provenance_issue(
            artifact,
            [read_text],
            require_answer_binding=True,
            original_question=question,
            qa_memory_relevant_memory_ids=[memory_id],
            qa_memory_expected_source_task_ids=["triviaqa:different"],
        )
        self.assertIsNotNone(wrong_source)

    def test_exact_source_qamemory_overrides_wh_field_only_for_matching_source(
        self,
    ) -> None:
        question = "Which 1975 film starred Diana Ross?"
        evidence_span = "Diana Ross starred in the film Mahogany."
        memory_id = "qa-memory-mahogany"
        source_task_id = "triviaqa:tc_mahogany"
        artifact = json.dumps(
            {
                "question_scope": question,
                "answer_slot": {
                    "answer_type": "entity",
                    "answer_cardinality": "single",
                    "qualifiers": [],
                    "proposition_index": 0,
                    "answer_field": "object_or_attribute_value",
                },
                "evidence_propositions": [
                    {
                        "subject": "Diana Ross",
                        "relation": "starred in",
                        "object_or_attribute_value": "the film Mahogany",
                        "qualifiers": [],
                        "evidence_span": evidence_span,
                    }
                ],
                "multi_hop_chain": ["bind the film title"],
                "candidate_answer": "Mahogany",
                "evidence": [evidence_span],
            }
        )
        read_text = _ReadReceiptText(
            f"Question: {question}\nAnswer: {evidence_span}",
            tool_id="triviaqa.qa_memory",
            record_id=memory_id,
            paraphrase_question=question,
            paraphrase_answer_statement=evidence_span,
            source_train_task_id=source_task_id,
            canonical_answer="Mahogany",
            semantic_preserving_paraphrase=True,
        )
        exact = AgentWorkflowEnv._exact_source_qa_memory_canonical_answer(
            artifact,
            [read_text],
            qa_memory_relevant_memory_ids=[memory_id],
            qa_memory_expected_source_task_ids=[source_task_id],
        )
        self.assertEqual("Mahogany", exact)
        candidate, issue = AgentWorkflowEnv._reasoner_candidate(
            artifact,
            original_question=question,
            minimum_evidence_propositions=1,
            minimum_reasoning_steps=1,
            preserve_question_derived_answer_field=True,
            exact_source_canonical_answer=exact,
        )
        self.assertEqual("Mahogany", candidate)
        self.assertIsNone(issue)

        wrong_source_exact = (
            AgentWorkflowEnv._exact_source_qa_memory_canonical_answer(
                artifact,
                [read_text],
                qa_memory_relevant_memory_ids=[memory_id],
                qa_memory_expected_source_task_ids=["triviaqa:different"],
            )
        )
        self.assertIsNone(wrong_source_exact)
        candidate, issue = AgentWorkflowEnv._reasoner_candidate(
            artifact,
            original_question=question,
            minimum_evidence_propositions=1,
            minimum_reasoning_steps=1,
            preserve_question_derived_answer_field=True,
            exact_source_canonical_answer=wrong_source_exact,
        )
        self.assertIsNone(candidate)
        self.assertIn("overt wh-dependency", issue or "")

    def test_exact_source_qamemory_uses_paired_question_for_scope(self) -> None:
        question = "Which player finished in first place in the race?"
        paraphrase_question = "Who finished in first place in the race?"
        evidence_span = "Alice won the race."
        memory_id = "qa-memory-first-place"
        source_task_id = "triviaqa:tc_first_place"
        artifact = json.dumps(
            {
                "question_scope": question,
                "answer_slot": {
                    "answer_type": "entity",
                    "answer_cardinality": "single",
                    "qualifiers": ["first place"],
                    "proposition_index": 0,
                    "answer_field": "subject",
                },
                "evidence_propositions": [
                    {
                        "subject": "Alice",
                        "relation": "won",
                        "object_or_attribute_value": "the race",
                        "qualifiers": ["first place"],
                        "evidence_span": evidence_span,
                    }
                ],
                "multi_hop_chain": ["bind the first-place finisher"],
                "candidate_answer": "Alice",
                "evidence": [evidence_span],
            }
        )
        read_text = _ReadReceiptText(
            f"Question: {paraphrase_question}\nAnswer: {evidence_span}",
            tool_id="triviaqa.qa_memory",
            record_id=memory_id,
            paraphrase_question=paraphrase_question,
            paraphrase_answer_statement=evidence_span,
            source_train_task_id=source_task_id,
            canonical_answer="Alice",
            semantic_preserving_paraphrase=True,
        )

        self.assertIsNone(
            AgentWorkflowEnv._reasoner_evidence_provenance_issue(
                artifact,
                [read_text],
                require_answer_binding=True,
                original_question=question,
                qa_memory_relevant_memory_ids=[memory_id],
                qa_memory_expected_source_task_ids=[source_task_id],
            )
        )

    def test_exact_source_multi_part_canonical_answer_is_not_minimal_failure(
        self,
    ) -> None:
        canonical = "One trophy is gold; three trophies are silver"
        verifier = json.dumps(
            {
                "candidate_answer": canonical,
                "evidence_supported": True,
                "entity_attribute_binding_correct": True,
                "alias_binding_correct": True,
                "answer_type_cardinality_correct": True,
                "multi_hop_complete": True,
                "minimal_answer_surface": False,
                "scope_preserved": True,
                "verification_status": "repair_required",
            }
        )
        candidate, issue = AgentWorkflowEnv._verifier_candidate(verifier)
        self.assertIsNone(candidate)
        self.assertIn("minimal_answer_surface", issue or "")
        candidate, issue = AgentWorkflowEnv._verifier_candidate(
            verifier,
            exact_source_canonical_answer=canonical,
        )
        self.assertEqual(canonical, candidate)
        self.assertIsNone(issue)

    def test_failure_continuation_preserves_qamemory_query_task_id(self) -> None:
        graph = _trivia_semantic_graph()
        record = AgentFailureRecord(
            request_id="reader-continuation",
            agent_id="reader",
            phase=ExecutionPhase.SINGLE,
            graph_revision=graph.revision,
            error_type="ReactExecutionError",
            message="bounded retrieval continuation",
            metadata={
                "tool_receipts": [_test_read_receipt("public-read")],
                "qa_memory_query_task_id": "triviaqa:tc_public",
            },
        )
        continuation = AgentWorkflowEnv._failure_continuation_candidate(record)
        self.assertIsNotNone(continuation)
        assert continuation is not None
        self.assertEqual(
            "triviaqa:tc_public",
            continuation["qa_memory_query_task_id"],
        )

    def test_reasoner_entity_gate_accepts_only_receipt_title_bound_alias(
        self,
    ) -> None:
        question = "Where in England was Dame Judi Dench born?"
        evidence_span = (
            "Dench was born in Heworth, North Riding of Yorkshire."
        )
        artifact = json.dumps(
            {
                "question_scope": question,
                "answer_slot": {
                    "answer_type": "location",
                    "answer_cardinality": "single",
                    "qualifiers": ["in England"],
                    "proposition_index": 0,
                    "answer_field": "object_or_attribute_value",
                },
                "evidence_propositions": [
                    {
                        "subject": "Dench",
                        "relation": "was born in",
                        "object_or_attribute_value": (
                            "Heworth, North Riding of Yorkshire"
                        ),
                        "qualifiers": ["in England"],
                        "evidence_span": evidence_span,
                    }
                ],
                "multi_hop_chain": [
                    "bind Dench to the receipt-backed passage entity"
                ],
                "candidate_answer": "Heworth, North Riding of Yorkshire",
                "evidence": ["dench-birthplace"],
            }
        )

        def receipt(title: str) -> dict[str, object]:
            return {
                "tool_id": QA_RETRIEVAL_TOOL_ID,
                "tool_version": "test-v1",
                "request": {
                    "action": "read",
                    "arguments": {"passage_id": "dench-birthplace"},
                },
                "result": {
                    "value": {
                        "operation": "read",
                        "passage_id": "dench-birthplace",
                        "passage": {
                            "passage_id": "dench-birthplace",
                            "title": title,
                            "text": evidence_span,
                        },
                    },
                    "completed": True,
                },
                "error_type": None,
            }

        grounded_read = AgentWorkflowEnv._successful_read_text(
            receipt("Judi Dench"),
            QA_RETRIEVAL_TOOL_ID,
        )
        self.assertIsNotNone(grounded_read)
        assert grounded_read is not None
        self.assertIsNone(
            AgentWorkflowEnv._reasoner_evidence_provenance_issue(
                artifact,
                [grounded_read],
                require_answer_binding=True,
                original_question=question,
            )
        )

        wrong_title_read = AgentWorkflowEnv._successful_read_text(
            receipt("Judy Dench"),
            QA_RETRIEVAL_TOOL_ID,
        )
        self.assertIsNotNone(wrong_title_read)
        assert wrong_title_read is not None
        issue = AgentWorkflowEnv._reasoner_evidence_provenance_issue(
            artifact,
            [wrong_title_read],
            require_answer_binding=True,
            original_question=question,
        )
        self.assertIn("no deterministic entity binding", issue or "")

    def test_reasoner_entity_gate_requires_receipt_grounded_proposition_lineage(
        self,
    ) -> None:
        question = "Where in England was Dr Alice Carter born?"
        birth_span = "Carter was born in Oakfield, North County."
        containment_span = (
            "Oakfield is part of the city of Northbridge in England."
        )

        def read_text(title: str, text: str) -> str:
            receipt = {
                "tool_id": QA_RETRIEVAL_TOOL_ID,
                "tool_version": "test-v1",
                "request": {
                    "action": "read",
                    "arguments": {"passage_id": title.casefold()},
                },
                "result": {
                    "value": {
                        "operation": "read",
                        "passage_id": title.casefold(),
                        "passage": {
                            "passage_id": title.casefold(),
                            "title": title,
                            "text": text,
                        },
                    },
                    "completed": True,
                },
                "error_type": None,
            }
            value = AgentWorkflowEnv._successful_read_text(
                receipt,
                QA_RETRIEVAL_TOOL_ID,
            )
            self.assertIsNotNone(value)
            assert value is not None
            return value

        def artifact(
            propositions: list[dict[str, object]],
            *,
            proposition_index: int,
            answer_field: str,
            candidate_answer: str,
        ) -> str:
            return json.dumps(
                {
                    "question_scope": question,
                    "answer_slot": {
                        "answer_type": "location",
                        "answer_cardinality": "single",
                        "qualifiers": ["in England"],
                        "proposition_index": proposition_index,
                        "answer_field": answer_field,
                    },
                    "evidence_propositions": propositions,
                    "multi_hop_chain": [
                        "bind the question entity to the birthplace",
                        "resolve the birthplace locality",
                    ],
                    "candidate_answer": candidate_answer,
                    "evidence": [
                        proposition["evidence_span"]
                        for proposition in propositions
                    ],
                }
            )

        malformed = artifact(
            [
                {
                    "subject": "Oakfield",
                    "relation": "was born in",
                    "object_or_attribute_value": "North County",
                    "qualifiers": ["in England"],
                    "evidence_span": birth_span,
                }
            ],
            proposition_index=0,
            answer_field="subject",
            candidate_answer="Oakfield",
        )
        malformed_issue = AgentWorkflowEnv._reasoner_evidence_provenance_issue(
            malformed,
            [
                read_text("Alice Carter", "A profile of Dr Alice Carter."),
                read_text("Oakfield", birth_span),
            ],
            require_answer_binding=True,
            original_question=question,
        )
        self.assertIn("no deterministic entity binding", malformed_issue or "")

        first_hop = {
            "subject": "Carter",
            "relation": "was born in",
            "object_or_attribute_value": "Oakfield, North County",
            "qualifiers": ["in England"],
            "evidence_span": birth_span,
        }
        direct = artifact(
            [first_hop],
            proposition_index=0,
            answer_field="object_or_attribute_value",
            candidate_answer="Oakfield, North County",
        )
        birth_read = read_text("Alice Carter", birth_span)
        self.assertIsNone(
            AgentWorkflowEnv._reasoner_evidence_provenance_issue(
                direct,
                [birth_read],
                require_answer_binding=True,
                original_question=question,
            )
        )

        second_hop = {
            "subject": "Oakfield",
            "relation": "is part of the city of",
            "object_or_attribute_value": "Northbridge",
            "qualifiers": ["in England"],
            "evidence_span": containment_span,
        }
        two_hop = artifact(
            [first_hop, second_hop],
            proposition_index=1,
            answer_field="object_or_attribute_value",
            candidate_answer="Northbridge",
        )
        self.assertIsNone(
            AgentWorkflowEnv._reasoner_evidence_provenance_issue(
                two_hop,
                [birth_read, read_text("Oakfield", containment_span)],
                require_answer_binding=True,
                original_question=question,
            )
        )

    def test_reasoner_provenance_rejects_fabricated_bridge_proposition(
        self,
    ) -> None:
        question = (
            "Which company acquired the business founded by Dr Alice Carter?"
        )
        founder_span = "Dr Alice Carter founded RealCo."
        acquisition_span = "Widget Holdings acquired RealCo."

        def read_text(passage_id: str, title: str, text: str) -> str:
            receipt = {
                "tool_id": QA_RETRIEVAL_TOOL_ID,
                "tool_version": "test-v1",
                "request": {
                    "action": "read",
                    "arguments": {"passage_id": passage_id},
                },
                "result": {
                    "value": {
                        "operation": "read",
                        "passage_id": passage_id,
                        "passage": {
                            "passage_id": passage_id,
                            "title": title,
                            "text": text,
                        },
                    },
                    "completed": True,
                },
                "error_type": None,
            }
            value = AgentWorkflowEnv._successful_read_text(
                receipt,
                QA_RETRIEVAL_TOOL_ID,
            )
            self.assertIsNotNone(value)
            assert value is not None
            return value

        def artifact(bridge_company: str) -> str:
            return json.dumps(
                {
                    "question_scope": question,
                    "answer_slot": {
                        "answer_type": "entity",
                        "answer_cardinality": "single",
                        "qualifiers": [],
                        "proposition_index": 1,
                        "answer_field": "subject",
                    },
                    "evidence_propositions": [
                        {
                            "subject": "Dr Alice Carter",
                            "relation": "founded",
                            "object_or_attribute_value": bridge_company,
                            "qualifiers": [],
                            "evidence_span": founder_span,
                        },
                        {
                            "subject": "Widget Holdings",
                            "relation": "acquired",
                            "object_or_attribute_value": bridge_company,
                            "qualifiers": [],
                            "evidence_span": acquisition_span.replace(
                                "RealCo",
                                bridge_company,
                            ),
                        },
                    ],
                    "multi_hop_chain": [
                        "Dr Alice Carter founded the business",
                        "Widget Holdings acquired that business",
                    ],
                    "candidate_answer": "Widget Holdings",
                    "evidence": [founder_span, acquisition_span],
                }
            )

        reads = [
            read_text("founder", "Dr Alice Carter", founder_span),
            read_text("acquisition", "RealCo", acquisition_span),
        ]
        self.assertIsNone(
            AgentWorkflowEnv._reasoner_evidence_provenance_issue(
                artifact("RealCo"),
                reads,
                require_answer_binding=True,
                original_question=question,
            )
        )
        issue = AgentWorkflowEnv._reasoner_evidence_provenance_issue(
            artifact("FakeCo"),
            reads,
            require_answer_binding=True,
            original_question=question,
        )
        self.assertIn(
            "evidence_propositions[0].object_or_attribute_value",
            issue or "",
        )

    def test_reasoner_subject_answer_binds_question_event_argument(self) -> None:
        question = "Who wrote The Trial?"

        def issue_for(evidence_span: str, subject: str, work: str) -> str | None:
            passage_id = "work"
            receipt = {
                "tool_id": QA_RETRIEVAL_TOOL_ID,
                "tool_version": "test-v1",
                "request": {
                    "action": "read",
                    "arguments": {"passage_id": passage_id},
                },
                "result": {
                    "value": {
                        "operation": "read",
                        "passage_id": passage_id,
                        "passage": {
                            "passage_id": passage_id,
                            "title": work,
                            "text": evidence_span,
                        },
                    },
                    "completed": True,
                },
                "error_type": None,
            }
            read_text = AgentWorkflowEnv._successful_read_text(
                receipt,
                QA_RETRIEVAL_TOOL_ID,
            )
            self.assertIsNotNone(read_text)
            assert read_text is not None
            artifact = json.dumps(
                {
                    "question_scope": question,
                    "answer_slot": {
                        "answer_type": "entity",
                        "answer_cardinality": "single",
                        "qualifiers": [],
                        "proposition_index": 0,
                        "answer_field": "subject",
                    },
                    "evidence_propositions": [
                        {
                            "subject": subject,
                            "relation": "wrote",
                            "object_or_attribute_value": work,
                            "qualifiers": [],
                            "evidence_span": evidence_span,
                        }
                    ],
                    "multi_hop_chain": [f"{subject} wrote {work}"],
                    "candidate_answer": subject,
                    "evidence": [evidence_span],
                }
            )
            return AgentWorkflowEnv._reasoner_evidence_provenance_issue(
                artifact,
                [read_text],
                require_answer_binding=True,
                original_question=question,
            )

        self.assertIsNone(
            issue_for("Franz Kafka wrote The Trial.", "Franz Kafka", "The Trial")
        )
        issue = issue_for(
            "Jane Doe wrote Other Book. The Trial was published later.",
            "Jane Doe",
            "Other Book",
        )
        self.assertIn("non-answer proposition argument", issue or "")

    def test_successful_read_receipt_requires_completed_matching_passage_ids(
        self,
    ) -> None:
        receipt = _test_read_receipt("p1")
        self.assertTrue(
            AgentWorkflowEnv._successful_read_receipt(
                receipt,
                QA_RETRIEVAL_TOOL_ID,
            )
        )
        for name, mutate in (
            (
                "incomplete",
                lambda value: value["result"].__setitem__("completed", False),
            ),
            (
                "top-level id mismatch",
                lambda value: value["result"]["value"].__setitem__(
                    "passage_id",
                    "p2",
                ),
            ),
            (
                "nested id mismatch",
                lambda value: value["result"]["value"]["passage"].__setitem__(
                    "passage_id",
                    "p2",
                ),
            ),
        ):
            with self.subTest(name=name):
                invalid = json.loads(json.dumps(receipt))
                mutate(invalid)
                self.assertFalse(
                    AgentWorkflowEnv._successful_read_receipt(
                        invalid,
                        QA_RETRIEVAL_TOOL_ID,
                    )
                )

    def test_verifier_repair_diagnosis_attributes_false_verdict_upstream(
        self,
    ) -> None:
        verifier_artifact = (
            "Candidate answer: 1960s\n"
            "Evidence supported: false\n"
            "Entity attribute binding correct: false\n"
            "Alias binding correct: true\n"
            "Answer type cardinality correct: true\n"
            "Multi-hop complete: false\n"
            "Minimal answer surface: true\n"
            "Scope preserved: false\n"
            "Verification status: repair_required\n"
            "Reasoning: the selected event does not preserve the question scope"
        )
        candidate, issue = AgentWorkflowEnv._verifier_candidate(
            verifier_artifact
        )
        self.assertIsNone(candidate)
        self.assertEqual(
            "Verifier field 'evidence_supported' must be true",
            issue,
        )

        registry = make_registry()
        env = AgentWorkflowEnv(
            registry,
            runtime=_trivia_semantic_runtime(registry, _ImmediateGateway()),
            graph=_trivia_semantic_graph(),
            problem="Who wrote the novel?",
            semantic_protocol=QA_VERIFIED_ANSWER_LINEAGE_PROTOCOL,
            recovery_policy="preserve_diagnose_repair_augment",
            required_evidence_tool_id=QA_RETRIEVAL_TOOL_ID,
        )
        attribution = env._semantic_repair_attribution(
            f"Verifier 'verifier' semantic artifact is invalid: {issue}"
        )
        self.assertIsNotNone(attribution)
        assert attribution is not None
        self.assertEqual("reasoner", attribution["responsible_agent_id"])
        self.assertEqual(
            "reasoner_semantic_artifact",
            attribution["responsible_constraint"],
        )

    def test_verifier_requires_exact_reasoner_read_receipt_lineage(self) -> None:
        registry = make_registry()
        env = AgentWorkflowEnv(
            registry,
            runtime=_hotpot_semantic_runtime(registry, _ImmediateGateway()),
            graph=_hotpot_semantic_graph(),
            problem="What is the capital of France?",
            semantic_protocol="hotpotqa_verified_answer_slot_v1",
            recovery_policy="preserve_diagnose_repair_augment",
            required_evidence_tool_id=QA_RETRIEVAL_TOOL_ID,
        )
        receipt = _test_read_receipt("lineage-passage")
        metadata = {
            "reasoner": {
                "artifact_version": "reasoner:v1",
                "tool_receipts": [receipt],
            },
            "verifier": {
                "input_artifact_provenance": [
                    {
                        "source_agent_id": "reasoner",
                        "artifact_version": "reasoner:v1",
                        "tool_receipts": [],
                    }
                ]
            },
        }
        texts, issue = env._verifier_read_receipt_lineage(
            metadata,
            reasoner_id="reasoner",
            verifier_id="verifier",
        )
        self.assertIsNone(issue)
        self.assertEqual(("Paris is the capital of France.",), texts)

        metadata["verifier"]["input_artifact_provenance"][0][
            "artifact_version"
        ] = "reasoner:stale"
        texts, issue = env._verifier_read_receipt_lineage(
            metadata,
            reasoner_id="reasoner",
            verifier_id="verifier",
        )
        self.assertEqual((), texts)
        self.assertIn("current Reasoner", issue or "")

    async def test_unchanged_semantic_failure_exhausts_one_modify_then_augments(
        self,
    ) -> None:
        registry = make_registry()
        gateway = _HotpotSemanticGateway(verifier_supported=False)
        env = AgentWorkflowEnv(
            registry,
            runtime=_hotpot_semantic_runtime(registry, gateway),
            graph=_hotpot_semantic_graph(),
            problem="What is the capital of France?",
            execute_on_edit=True,
            semantic_protocol="hotpotqa_verified_answer_slot_v1",
            recovery_policy="preserve_diagnose_repair_augment",
            required_evidence_tool_id=QA_RETRIEVAL_TOOL_ID,
        )
        initial = await env.step(
            '{"action":"modify_agent","agent_id":"reader",'
            '"contract":"read grounded database evidence"}'
        )
        self.assertTrue(initial.accepted, initial.feedback)
        preserved_reader_artifact = env._progressive_outputs["reader"]
        self.assertEqual(("reasoner",), env._mandatory_repair_agent_ids())
        self.assertEqual(("modify_agent",), env.model_admissible_action_types())

        repaired = await env.step(
            '{"action":"modify_agent","agent_id":"reasoner",'
            '"contract":"align the same grounded evidence to the answer slot"}'
        )

        self.assertTrue(repaired.accepted, repaired.feedback)
        self.assertEqual(
            preserved_reader_artifact,
            env._progressive_outputs["reader"],
        )
        self.assertIn("reasoner", env._repair_exhausted_agent_ids)
        self.assertEqual((), env._mandatory_repair_agent_ids())
        self.assertEqual(("add_subgraph",), env.model_admissible_action_types())
        add_domain = env.model_admissible_action_targets()["add_subgraph"]
        self.assertEqual(
            ["evidence_retriever"],
            add_domain["admitted_new_role_families"],
        )
        self.assertEqual(2, len(env.history))

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
                    "relation": "is a country in",
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

        decade = json.loads(json.dumps(artifact))
        decade_question = (
            "In which decade did Billboard magazine first publish a hit chart?"
        )
        decade["question_scope"] = decade_question
        decade["answer_slot"]["answer_type"] = "date"
        decade["evidence_propositions"][0].update(
            {
                "subject": "Billboard magazine",
                "relation": "published its first music hit parade on",
                "object_or_attribute_value": "January 4, 1936",
                "evidence_span": (
                    "On January 4, 1936, Billboard magazine published its first "
                    "music hit parade."
                ),
            }
        )
        decade["candidate_answer"] = "30s"
        candidate, issue = env._reasoner_candidate(
            json.dumps(decade),
            original_question=decade_question,
        )
        self.assertEqual("30s", candidate)
        self.assertIsNone(issue)
        provenance_issue = env._reasoner_evidence_provenance_issue(
            json.dumps(decade),
            [
                decade["evidence_propositions"][0]["evidence_span"],
                decade["evidence_propositions"][1]["evidence_span"],
            ],
            require_answer_binding=True,
            original_question=decade_question,
        )
        self.assertIsNone(provenance_issue)

        wrong_decade = json.loads(json.dumps(decade))
        wrong_decade["candidate_answer"] = "40s"
        candidate, issue = env._reasoner_candidate(
            json.dumps(wrong_decade),
            original_question=decade_question,
        )
        self.assertIsNone(candidate)
        self.assertIn("verified year-to-decade", str(issue))

    def test_reasoner_preserves_wh_dependency_and_question_relation(self) -> None:
        question = (
            "Which coastal-born novelist won the international fiction prize "
            "in 1995?"
        )
        evidence_span = (
            "Avery Morgan was a coastal-born novelist. In 1995, Avery Morgan "
            "received the international fiction prize."
        )
        artifact = {
            "question_scope": question,
            "answer_slot": {
                "answer_type": "entity",
                "answer_cardinality": "single",
                "qualifiers": ["coastal-born", "1995"],
                "proposition_index": 0,
                "answer_field": "subject",
            },
            "evidence_propositions": [
                {
                    "subject": "Avery Morgan",
                    "relation": "received",
                    "object_or_attribute_value": (
                        "the international fiction prize"
                    ),
                    "qualifiers": ["in 1995"],
                    "evidence_span": evidence_span,
                }
            ],
            "multi_hop_chain": [
                "Avery Morgan --received in 1995--> international fiction prize"
            ],
            "candidate_answer": "Avery Morgan",
            "evidence": [evidence_span],
        }

        candidate, issue = AgentWorkflowEnv._reasoner_candidate(
            json.dumps(artifact),
            original_question=question,
            minimum_evidence_propositions=1,
            minimum_reasoning_steps=1,
        )
        self.assertEqual("Avery Morgan", candidate)
        self.assertIsNone(issue)

        misbound = json.loads(json.dumps(artifact))
        misbound["answer_slot"]["answer_field"] = "object_or_attribute_value"
        misbound["candidate_answer"] = "the international fiction prize"
        candidate, issue = AgentWorkflowEnv._reasoner_candidate(
            json.dumps(misbound),
            original_question=question,
            minimum_evidence_propositions=1,
            minimum_reasoning_steps=1,
        )
        self.assertIsNone(candidate)
        self.assertIn("overt wh-dependency", str(issue))

        fixed_slot_alternate = json.loads(json.dumps(artifact))
        fixed_slot_alternate["evidence_propositions"][0]["subject"] = (
            "the international fiction prize"
        )
        fixed_slot_alternate["evidence_propositions"][0][
            "object_or_attribute_value"
        ] = "Avery Morgan"
        candidate, issue = AgentWorkflowEnv._reasoner_candidate(
            json.dumps(fixed_slot_alternate),
            original_question=question,
            minimum_evidence_propositions=1,
            minimum_reasoning_steps=1,
            preserve_question_derived_answer_field=True,
        )
        self.assertIsNone(candidate)
        self.assertIn("alternate proposition argument", str(issue))
        self.assertIn("overt wh-dependency", str(issue))
        self.assertIn("preserve candidate_answer and answer_slot", str(issue))
        self.assertNotIn("set answer_field", str(issue))
        self.assertNotIn("set answer_slot.answer_field", str(issue))

        unrelated_relation = json.loads(json.dumps(artifact))
        unrelated_relation["evidence_propositions"][0]["relation"] = "was"
        provenance_issue = AgentWorkflowEnv._reasoner_evidence_provenance_issue(
            json.dumps(unrelated_relation),
            [evidence_span],
            require_answer_binding=True,
            original_question=question,
        )
        self.assertIn("requested relation", str(provenance_issue))

    def test_reasoner_rejects_duplicate_subject_object_answer_binding(self) -> None:
        question = "What is the capital of France?"
        artifact = {
            "question_scope": question,
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
                    "relation": "is a country in",
                    "object_or_attribute_value": "Europe",
                    "qualifiers": [],
                    "evidence_span": "France is a country in Europe.",
                },
            ],
            "multi_hop_chain": ["bind France", "select its capital"],
            "candidate_answer": "Paris",
            "evidence": ["Paris is the capital of France."],
        }

        candidate, issue = AgentWorkflowEnv._reasoner_candidate(
            json.dumps(artifact),
            original_question=question,
        )
        self.assertEqual("Paris", candidate)
        self.assertIsNone(issue)

        duplicate_binding = json.loads(json.dumps(artifact))
        duplicate_binding["evidence_propositions"][0]["subject"] = "Paris"
        candidate, issue = AgentWorkflowEnv._reasoner_candidate(
            json.dumps(duplicate_binding),
            original_question=question,
        )
        self.assertIsNone(candidate)
        self.assertIn("distinct subject and object_or_attribute_value", issue or "")
        self.assertIn("self-reported entity binding", issue or "")
        self.assertIn("evidence_propositions[0]", issue or "")
        self.assertIn("fixed answer field 'object_or_attribute_value'", issue or "")
        self.assertIn("repair only the non-answer field 'subject'", issue or "")

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

    async def test_retriever_reasoner_relation_direction_is_semantic(self) -> None:
        graph = AgentGraph(
            [
                AgentNode(
                    "reader",
                    "cheap",
                    "retrieve evidence",
                    role_family="evidence_retriever",
                    artifact_type="retrieval_evidence",
                ),
                AgentNode(
                    "reasoner",
                    "balanced",
                    "bind evidence to the answer slot",
                    role_family="reasoner",
                    allowed_tools=(QA_RETRIEVAL_TOOL_ID,),
                    execution_mode="react",
                    artifact_type="semantic_candidate",
                ),
            ]
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
            required_evidence_tool_id=QA_RETRIEVAL_TOOL_ID,
        )

        candidates = env._all_model_admissible_relation_candidates()
        self.assertIn(
            {
                "source_id": "reader",
                "target_id": "reasoner",
                "source_to_target": True,
                "target_to_source": False,
            },
            candidates,
        )
        self.assertNotIn(
            {
                "source_id": "reasoner",
                "target_id": "reader",
                "source_to_target": True,
                "target_to_source": False,
            },
            candidates,
        )
        self.assertIn(
            {
                "source_id": "reader",
                "target_id": "reasoner",
                "source_to_target": True,
                "target_to_source": True,
            },
            candidates,
        )

        revision = env.revision
        reverse_only = await env.step(
            '{"action":"set_relation","source_id":"reasoner",'
            '"target_id":"reader","source_to_target":true,'
            '"target_to_source":false}'
        )
        self.assertFalse(reverse_only.accepted)
        self.assertIn("invalid one-way semantic handoff", reverse_only.feedback)
        self.assertEqual(revision, env.revision)

        forward = await env.step(
            '{"action":"set_relation","source_id":"reader",'
            '"target_id":"reasoner","source_to_target":true,'
            '"target_to_source":false}'
        )
        self.assertTrue(forward.accepted, forward.feedback)
        reciprocal = await env.step(
            '{"action":"set_relation","source_id":"reader",'
            '"target_id":"reasoner","source_to_target":true,'
            '"target_to_source":true}'
        )
        self.assertTrue(reciprocal.accepted, reciprocal.feedback)

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
            "Search for the birthplace with a limit of 3 results.",
            "Use an explicit locus symbol (e.g., :loc) in the search query.",
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
                    "Verify the candidate answer 'Meridian Archive' against "
                    "the routed evidence"
                ),
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

    async def test_triviaqa_retriever_contract_rejects_question_external_literals(
        self,
    ) -> None:
        registry = make_registry()
        gateway = _ImmediateGateway()
        env = AgentWorkflowEnv(
            registry,
            runtime=_trivia_semantic_runtime(registry, gateway),
            problem=(
                "In which decade did Billboard magazine first publish an "
                "American hit chart?"
            ),
            execute_on_edit=False,
            semantic_protocol=QA_VERIFIED_ANSWER_LINEAGE_PROTOCOL,
            recovery_policy="preserve_diagnose_repair_augment",
            required_evidence_tool_id=QA_RETRIEVAL_TOOL_ID,
        )

        for narrowed_contract in (
            "Search for the Billboard Hot 100 as the first chart.",
            "Search the Billboard year-end chart while preserving the relation.",
            "Search for first publication in 1958 or 1959.",
            "Search for first publication in 1941.",
        ):
            revision = env.revision
            rejected = await env.step(
                json.dumps(
                    {
                        "action": "add_subgraph",
                        "agents": [
                            {
                                "agent_id": "retriever",
                                "model_id": "balanced",
                                "contract": narrowed_contract,
                                "role_family": "evidence_retriever",
                                "allowed_tools": [QA_RETRIEVAL_TOOL_ID],
                                "execution_mode": "react",
                            }
                        ],
                        "relations": [],
                    }
                )
            )
            self.assertFalse(rejected.accepted)
            self.assertIn(
                "question-external semantic literals",
                rejected.feedback,
            )
            self.assertEqual(revision, env.revision)
            self.assertFalse(env.graph.has_node("retriever"))

        revision = env.revision
        rejected_completion_literal = await env.step(
            json.dumps(
                {
                    "action": "add_subgraph",
                    "agents": [
                        {
                            "agent_id": "retriever",
                            "model_id": "balanced",
                            "contract": "Retrieve evidence for the requested relation",
                            "completion_condition": (
                                "Complete after establishing publication in 1941"
                            ),
                            "role_family": "evidence_retriever",
                            "allowed_tools": [QA_RETRIEVAL_TOOL_ID],
                            "execution_mode": "react",
                        }
                    ],
                    "relations": [],
                }
            )
        )
        self.assertFalse(rejected_completion_literal.accepted)
        self.assertIn(
            "question-external semantic literals",
            rejected_completion_literal.feedback,
        )
        self.assertEqual(revision, env.revision)

        neutral = await env.step(
            json.dumps(
                {
                    "action": "add_subgraph",
                    "agents": [
                        {
                            "agent_id": "retriever",
                            "model_id": "balanced",
                            "contract": (
                                "Use spelling normalization, alias expansion, entity "
                                "disambiguation, query rewriting, and larger top-k "
                                "while preserving the original entity, relation, "
                                "qualifiers, and answer type"
                            ),
                            "role_family": "evidence_retriever",
                            "allowed_tools": [QA_RETRIEVAL_TOOL_ID],
                            "execution_mode": "react",
                        }
                    ],
                    "relations": [],
                }
            )
        )
        self.assertTrue(neutral.accepted, neutral.feedback)

        structural_count = await env.step(
            json.dumps(
                {
                    "action": "modify_agent",
                    "agent_id": "retriever",
                    "completion_condition": (
                        "Complete after at least 2 evidence propositions and "
                        "2 successful read receipts preserve the requested relation"
                    ),
                }
            )
        )
        self.assertTrue(structural_count.accepted, structural_count.feedback)

        env._failure_continuations["retriever"] = {
            "tool_receipts": [
                {
                    "tool_id": QA_RETRIEVAL_TOOL_ID,
                    "tool_version": "test-v1",
                    "request": {
                        "action": "read",
                        "arguments": {"passage_id": "billboard-history"},
                    },
                    "result": {
                        "value": {
                            "operation": "read",
                            "passage_id": "billboard-history",
                            "passage": {
                                "passage_id": "billboard-history",
                                "title": "Billboard charts",
                                "text": (
                                    "The publication introduced another chart in "
                                    "1941."
                                ),
                            },
                        },
                        "completed": True,
                    },
                    "error_type": None,
                }
            ]
        }
        receipt_grounded = await env.step(
            json.dumps(
                {
                    "action": "modify_agent",
                    "agent_id": "retriever",
                    "completion_condition": (
                        "Complete only when a successful read receipt supports 1941"
                    ),
                }
            )
        )
        self.assertTrue(receipt_grounded.accepted, receipt_grounded.feedback)

        revision = env.revision
        rejected_modify = await env.step(
            json.dumps(
                {
                    "action": "modify_agent",
                    "agent_id": "retriever",
                    "contract": "Search the Hot 100 first published in 1958.",
                }
            )
        )
        self.assertFalse(rejected_modify.accepted)
        self.assertIn(
            "question-external semantic literals",
            rejected_modify.feedback,
        )
        self.assertEqual(revision, env.revision)
        self.assertEqual([], gateway.requests)

    async def test_triviaqa_contract_rejects_external_entity_without_losing_qualifier(
        self,
    ) -> None:
        registry = make_registry()
        env = AgentWorkflowEnv(
            registry,
            runtime=_trivia_semantic_runtime(
                registry,
                _ImmediateGateway(),
            ),
            problem=(
                "Which American-born Sinclair won the Nobel Prize for "
                "Literature in 1930?"
            ),
            execute_on_edit=False,
            semantic_protocol=QA_VERIFIED_ANSWER_LINEAGE_PROTOCOL,
            recovery_policy="preserve_diagnose_repair_augment",
            required_evidence_tool_id=QA_RETRIEVAL_TOOL_ID,
        )
        external_entity = await env.step(
            json.dumps(
                {
                    "action": "add_subgraph",
                    "agents": [
                        {
                            "agent_id": "retriever",
                            "model_id": "balanced",
                            "contract": (
                                "search for and read passages regarding "
                                "Nicolas Sinclair Nobel Prize"
                            ),
                            "role_family": "evidence_retriever",
                            "allowed_tools": [QA_RETRIEVAL_TOOL_ID],
                            "execution_mode": "react",
                        }
                    ],
                    "relations": [],
                }
            )
        )
        self.assertFalse(external_entity.accepted)
        self.assertIn("pre-execution obligations only", external_entity.feedback)
        self.assertFalse(env.graph.has_node("retriever"))

        faithful_scope = await env.step(
            json.dumps(
                {
                    "action": "add_subgraph",
                    "agents": [
                        {
                            "agent_id": "retriever",
                            "model_id": "balanced",
                            "contract": (
                                "find the American-born Nobel Prize winner in "
                                "Literature for 1930"
                            ),
                            "role_family": "evidence_retriever",
                            "allowed_tools": [QA_RETRIEVAL_TOOL_ID],
                            "execution_mode": "react",
                        }
                    ],
                    "relations": [],
                }
            )
        )
        self.assertTrue(faithful_scope.accepted, faithful_scope.feedback)

    async def test_triviaqa_contract_literals_ignore_short_evidence_fragments(
        self,
    ) -> None:
        registry = make_registry()
        env = AgentWorkflowEnv(
            registry,
            runtime=_trivia_semantic_runtime(
                registry,
                _ImmediateGateway(),
            ),
            problem=(
                "In which decade did Billboard magazine first publish an "
                "American hit chart?"
            ),
            execute_on_edit=False,
            semantic_protocol=QA_VERIFIED_ANSWER_LINEAGE_PROTOCOL,
            recovery_policy="preserve_diagnose_repair_augment",
            required_evidence_tool_id=QA_RETRIEVAL_TOOL_ID,
        )
        copied_evidence = (
            "The publication introduced its first national chart in spring"
        )
        passage_id = "atlas-dpr-wikipedia:000012167575"
        env._failure_continuations["retriever"] = {
            "rejected_completion": {
                "candidate_answer": "1940s",
                "evidence_span": "For the ",
                "evidence": [passage_id, copied_evidence],
            }
        }

        literals = env._public_semantic_contract_literals()

        self.assertIn("1940s", literals)
        self.assertIn(copied_evidence, literals)
        self.assertNotIn("For the", literals)
        self.assertNotIn(passage_id, literals)

        neutral = await env.step(
            json.dumps(
                {
                    "action": "add_subgraph",
                    "agents": [
                        {
                            "agent_id": "retriever",
                            "model_id": "balanced",
                            "contract": (
                                "Retrieve answer-free evidence for the question "
                                "entity and requested relation"
                            ),
                            "role_family": "evidence_retriever",
                            "allowed_tools": [QA_RETRIEVAL_TOOL_ID],
                            "execution_mode": "react",
                        }
                    ],
                    "relations": [],
                }
            )
        )
        self.assertTrue(neutral.accepted, neutral.feedback)

        revision = env.revision
        copied_candidate = await env.step(
            json.dumps(
                {
                    "action": "modify_agent",
                    "agent_id": "retriever",
                    "contract": "Return 1940s as the candidate answer",
                }
            )
        )
        self.assertFalse(copied_candidate.accepted)
        self.assertIn("pre-execution obligations only", copied_candidate.feedback)
        self.assertEqual(revision, env.revision)

        copied_span = await env.step(
            json.dumps(
                {
                    "action": "modify_agent",
                    "agent_id": "retriever",
                    "contract": f"Use {copied_evidence} as evidence",
                }
            )
        )
        self.assertFalse(copied_span.accepted)
        self.assertIn("pre-execution obligations only", copied_span.feedback)
        self.assertEqual(revision, env.revision)

        concrete_query_variation = await env.step(
            json.dumps(
                {
                    "action": "modify_agent",
                    "agent_id": "retriever",
                    "contract": (
                        "Iterate through query variations: 'Billboard first "
                        "chart' and 'Billboard inaugural chart'"
                    ),
                }
            )
        )
        self.assertFalse(concrete_query_variation.accepted)
        self.assertIn(
            "without concrete Tool invocation arguments",
            concrete_query_variation.feedback,
        )
        self.assertEqual(revision, env.revision)

        unsupported_canonical_literal = await env.step(
            json.dumps(
                {
                    "action": "modify_agent",
                    "agent_id": "retriever",
                    "contract": (
                        "Require canonical match for 'London' before completion"
                    ),
                }
            )
        )
        self.assertFalse(unsupported_canonical_literal.accepted)
        self.assertIn(
            "pre-execution obligations only",
            unsupported_canonical_literal.feedback,
        )
        self.assertEqual(revision, env.revision)

    async def test_triviaqa_contract_rejects_single_token_answer_assertion(
        self,
    ) -> None:
        registry = make_registry()
        env = AgentWorkflowEnv(
            registry,
            runtime=_trivia_semantic_runtime(
                registry,
                _ImmediateGateway(),
            ),
            problem="Where in England was Dame Judi Dench born?",
            execute_on_edit=False,
            semantic_protocol=QA_VERIFIED_ANSWER_LINEAGE_PROTOCOL,
            recovery_policy="preserve_diagnose_repair_augment",
            required_evidence_tool_id=QA_RETRIEVAL_TOOL_ID,
        )

        rejected = await env.step(
            json.dumps(
                {
                    "action": "add_subgraph",
                    "agents": [
                        {
                            "agent_id": "reasoner",
                            "model_id": "balanced",
                            "contract": (
                                "derive answer that Dame Judi Dench was born "
                                "in London from retrieved evidence"
                            ),
                            "role_family": "reasoner",
                            "allowed_tools": [QA_RETRIEVAL_TOOL_ID],
                            "execution_mode": "react",
                        }
                    ],
                    "relations": [],
                }
            )
        )

        self.assertFalse(rejected.accepted)
        self.assertIn("question-external single-token entity", rejected.feedback)
        self.assertIn("London", rejected.feedback)
        self.assertEqual((), env.graph.nodes)

        neutral = await env.step(
            json.dumps(
                {
                    "action": "add_subgraph",
                    "agents": [
                        {
                            "agent_id": "reasoner",
                            "model_id": "balanced",
                            "contract": (
                                "derive the answer in JSON by binding Dame Judi "
                                "Dench to receipt-grounded evidence"
                            ),
                            "role_family": "reasoner",
                            "allowed_tools": [QA_RETRIEVAL_TOOL_ID],
                            "execution_mode": "react",
                        }
                    ],
                    "relations": [],
                }
            )
        )

        self.assertTrue(neutral.accepted, neutral.feedback)

    async def test_triviaqa_contract_rejects_question_external_candidate_before_answer_marker(
        self,
    ) -> None:
        registry = make_registry()
        env = AgentWorkflowEnv(
            registry,
            runtime=_trivia_semantic_runtime(
                registry,
                _ImmediateGateway(),
            ),
            problem="Where in England was Dame Judi Dench born?",
            execute_on_edit=False,
            semantic_protocol=QA_VERIFIED_ANSWER_LINEAGE_PROTOCOL,
            recovery_policy="preserve_diagnose_repair_augment",
            required_evidence_tool_id=QA_RETRIEVAL_TOOL_ID,
        )

        rejected = await env.step(
            json.dumps(
                {
                    "action": "add_subgraph",
                    "agents": [
                        {
                            "agent_id": "reasoner",
                            "model_id": "balanced",
                            "contract": (
                                "Reason over receipt-grounded evidence, extracting "
                                "the specific geographic location (Downing Street, "
                                "Smith Square, Greater London) as the answer."
                            ),
                            "role_family": "reasoner",
                            "allowed_tools": [QA_RETRIEVAL_TOOL_ID],
                            "execution_mode": "react",
                        }
                    ],
                    "relations": [],
                }
            )
        )

        self.assertFalse(rejected.accepted)
        self.assertIn("pre-execution obligations only", rejected.feedback)
        self.assertEqual((), env.graph.nodes)

    async def test_triviaqa_reasoner_contract_rejects_bare_foreign_role_labels(
        self,
    ) -> None:
        for conflicting_contract in (
            "format",
            "retrieval",
            (
                "ground answer-free evidence for the original entity and "
                "requested relation in matching successful read Tool receipts"
            ),
        ):
            registry = make_registry()
            gateway = _ImmediateGateway()
            env = AgentWorkflowEnv(
                registry,
                runtime=_trivia_semantic_runtime(registry, gateway),
                problem="Which American author won the prize?",
                execute_on_edit=False,
                require_format_agent=True,
                semantic_protocol=QA_VERIFIED_ANSWER_LINEAGE_PROTOCOL,
                recovery_policy="preserve_diagnose_repair_augment",
                required_evidence_tool_id=QA_RETRIEVAL_TOOL_ID,
            )
            revision = env.revision

            rejected = await env.step(
                json.dumps(
                    {
                        "action": "add_subgraph",
                        "agents": [
                            {
                                "agent_id": "reasoner",
                                "model_id": "balanced",
                                "contract": conflicting_contract,
                                "role_family": "reasoner",
                                "allowed_tools": [QA_RETRIEVAL_TOOL_ID],
                                "execution_mode": "react",
                            }
                        ],
                        "relations": [],
                    }
                )
            )

            self.assertFalse(rejected.accepted)
            self.assertIn("Reasoner Agent", rejected.feedback)
            self.assertIn("another role responsibility", rejected.feedback)
            self.assertEqual(revision, env.revision)
            self.assertEqual((), env.graph.nodes)
            self.assertEqual([], gateway.requests)

    async def test_triviaqa_reasoner_contract_modify_preserves_role_boundary(
        self,
    ) -> None:
        registry = make_registry()
        gateway = _ImmediateGateway()
        env = AgentWorkflowEnv(
            registry,
            runtime=_trivia_semantic_runtime(registry, gateway),
            graph=_trivia_semantic_graph(),
            problem="Which American author won the prize?",
            execute_on_edit=False,
            require_format_agent=True,
            semantic_protocol=QA_VERIFIED_ANSWER_LINEAGE_PROTOCOL,
            recovery_policy="preserve_diagnose_repair_augment",
            required_evidence_tool_id=QA_RETRIEVAL_TOOL_ID,
        )
        original = env.graph.get_node("reasoner")

        for conflicting_contract in (
            "format",
            "retrieval",
            (
                "ground answer-free evidence for the original entity and "
                "requested relation in matching successful read Tool receipts"
            ),
        ):
            revision = env.revision
            rejected = await env.step(
                json.dumps(
                    {
                        "action": "modify_agent",
                        "agent_id": "reasoner",
                        "contract": conflicting_contract,
                    }
                )
            )
            self.assertFalse(rejected.accepted)
            self.assertIn("another role responsibility", rejected.feedback)
            self.assertEqual(revision, env.revision)
            current = env.graph.get_node("reasoner")
            self.assertEqual(original.contract, current.contract)
            self.assertEqual(original.role_family, current.role_family)
            self.assertEqual(original.execution_mode, current.execution_mode)
            self.assertEqual(original.allowed_tools, current.allowed_tools)
            self.assertEqual([], gateway.requests)

        repaired_contract = (
            "bind receipt-grounded evidence propositions to the original entity, "
            "requested relation, qualifiers, and answer slot; derive one semantic "
            "candidate without serializing the answer wrapper"
        )
        accepted = await env.step(
            json.dumps(
                {
                    "action": "modify_agent",
                    "agent_id": "reasoner",
                    "contract": repaired_contract,
                }
            )
        )
        self.assertTrue(accepted.accepted, accepted.feedback)
        current = env.graph.get_node("reasoner")
        self.assertEqual(repaired_contract, current.contract)
        self.assertEqual("reasoner", current.role_family)
        self.assertEqual("react", current.execution_mode.value)
        self.assertEqual((QA_RETRIEVAL_TOOL_ID,), current.allowed_tools)
        self.assertEqual([], gateway.requests)

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
                    "relation": "is a country in",
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
                    "request": {
                        "action": "read",
                        "arguments": {"passage_id": "p1"},
                    },
                    "result": {
                        "value": {
                            "operation": "read",
                            "passage_id": "p1",
                            "passage": {
                                "passage_id": "p1",
                                "text": (
                                    "Paris is the capital of France. "
                                    "France is a country in Europe."
                                )
                            },
                        },
                        "completed": True,
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
                        "relation": "is a country in",
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
                "reader": _test_evidence_retriever_artifact("p1"),
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
                            "passage_id": "p1",
                            "passage": {
                                "passage_id": "p1",
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
        env._progressive_output_metadata.update(
            {
                "reasoner": {
                    "artifact_version": "fixture:reasoner:current",
                    "input_artifact_versions": {},
                },
                "verifier": {
                    "artifact_version": "fixture:verifier:current",
                    "input_artifact_versions": {
                        "reasoner": "fixture:reasoner:current"
                    },
                },
                "formatter": {
                    "artifact_version": "fixture:formatter:current",
                    "input_artifact_versions": {
                        "verifier": "fixture:verifier:current"
                    },
                },
            }
        )

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
