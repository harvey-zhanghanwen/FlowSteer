from __future__ import annotations

import asyncio
from dataclasses import dataclass
import importlib.util
import json
from pathlib import Path
import unittest

import yaml

from src.interactive.agent_graph import AgentNode
from src.interactive.agent_runtime import (
    AgentRequest,
    ExecutionPhase,
    UpstreamMessage,
)
from src.interactive.hotpotqa_embedding_tool import (
    HOTPOTQA_QA_MEMORY_TOOL_ID,
    HotpotQAEmbeddingReactExecutionAdapter,
    build_hotpotqa_embedding_tool_registry,
)
from src.interactive.hotpotqa_transductive_qa_memory_index import (
    TRANSDUCTIVE_QA_MEMORY_CORPUS_VERSION,
)
from src.interactive.model_registry import ModelSpec, ProviderSpec
from src.interactive.openai_gateway import build_agent_messages
from src.interactive.tool_runtime import ToolRequest


ROOT = Path(__file__).resolve().parents[2]


def _load_script(name: str):
    path = ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_RUNNER = _load_script("evaluate_hotpotqa_round")


@dataclass(frozen=True)
class _Manifest:
    schema_version: str = "flowsteer.hotpotqa.transductive_qa_memory_index.v1"
    index_id: str = "hotpotqa-transductive-test-v1"
    corpus_version: str = TRANSDUCTIVE_QA_MEMORY_CORPUS_VERSION
    source: str = "HotpotQA 512+128"
    source_splits: tuple[str, ...] = ("train", "frozen_validation")
    embedding_model: str = "test-encoder"
    embedding_dimension: int = 3
    normalized: bool = True
    similarity: str = "cosine"
    frozen_top_k: int = 2
    source_record_count: int = 640
    source_train_count: int = 512
    source_evaluation_count: int = 128
    unique_source_count: int = 640
    cycled_record_count: int = 0
    paraphrase_count: int = 640
    frozen_validation_count: int = 128
    evaluation_overlap_count: int = 128
    contains_evaluation_answers: bool = True
    evaluation_regime: str = "transductive_retrieval"
    official_heldout_eligible: bool = False
    paraphrase_versions: tuple[str, ...] = ("transductive-v1",)
    paraphrase_provenances: tuple[str, ...] = ("offline-transductive",)


@dataclass(frozen=True)
class _Hit:
    memory_id: str
    source_train_task_id: str
    paraphrase_question: str
    paraphrase_answer_statement: str
    similarity: float
    rank: int


@dataclass(frozen=True)
class _Memory:
    memory_id: str
    source_train_task_id: str
    base_task_id: str
    cycled: bool
    paraphrase_question: str
    paraphrase_answer_statement: str
    canonical_answer: str
    paraphrase_version: str
    paraphrase_provenance: str


class _Index:
    manifest = _Manifest()

    def search(self, query: str, k: int) -> tuple[_Hit, ...]:
        del query, k
        return (
            _Hit("m1", "hotpotqa:train", "Alpha?", "Alpha is Ada.", 0.9, 1),
            _Hit(
                "m2",
                "hotpotqa:evaluation",
                "Validation alpha?",
                "Validation alpha is Ada.",
                0.8,
                2,
            ),
        )

    def read(self, memory_id: str) -> _Memory:
        return _Memory(
            memory_id,
            "hotpotqa:evaluation" if memory_id == "m2" else "hotpotqa:train",
            "hotpotqa:evaluation" if memory_id == "m2" else "hotpotqa:train",
            False,
            "Validation alpha?" if memory_id == "m2" else "Alpha?",
            "Validation alpha is Ada." if memory_id == "m2" else "Alpha is Ada.",
            "Ada",
            "transductive-v1",
            "offline-transductive",
        )


class _Gateway:
    async def generate(self, request):  # pragma: no cover - never invoked
        raise AssertionError(f"unexpected model call: {request}")


class HotpotQARound01TransductiveQAMemoryV19ProfileTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.best = yaml.safe_load(
            (ROOT / "config/evaluation_hotpotqa_round_01.yaml").read_text(
                encoding="utf-8"
            )
        )
        cls.v17 = yaml.safe_load(
            (
                ROOT / "config/evaluation_hotpotqa_round01_qa_memory_v17.yaml"
            ).read_text(encoding="utf-8")
        )
        cls.candidate = yaml.safe_load(
            (
                ROOT
                / "config/evaluation_hotpotqa_round01_transductive_qa_memory_v19.yaml"
            ).read_text(encoding="utf-8")
        )

    def test_round01_and_v17_architecture_boundaries_are_preserved(self) -> None:
        self.assertEqual(self.best["experiment"]["seed"], self.candidate["experiment"]["seed"])
        self.assertEqual(self.v17["director"], self.candidate["director"])
        candidate_graph = dict(self.candidate["agent_graph"])
        self.assertEqual(
            "preserve_diagnose_repair_augment",
            candidate_graph.pop("recovery_policy"),
        )
        self.assertEqual(self.v17["agent_graph"], candidate_graph)
        for section in (
            "evaluation",
            "grpo",
            "policy_sync",
            "exploration",
            "skills",
            "gpu",
            "deployment",
        ):
            self.assertEqual(self.v17[section], self.candidate[section])
        self.assertEqual(
            [
                "add_agent",
                "modify_agent",
                "delete_agent",
                "set_relation",
                "set_output",
                "finish",
            ],
            self.candidate["agent_graph"]["actions"],
        )
        self.assertNotIn("terminal_protocol", self.candidate["agent_graph"])
        self.assertNotIn("require_format_agent", self.candidate["agent_graph"])
        _RUNNER.validate_hotpot_config(self.candidate)

    def test_transductive_manifest_and_config_are_explicitly_ineligible(self) -> None:
        retrieval = self.candidate["qa_embedding_retrieval"]
        self.assertEqual("transductive_qa_memory", retrieval["corpus_kind"])
        self.assertEqual(640, retrieval["source_record_count"])
        self.assertEqual(512, retrieval["train_sample_count"])
        self.assertEqual(128, retrieval["validation_sample_count"])
        self.assertEqual(128, retrieval["evaluation_overlap_count"])
        self.assertTrue(retrieval["contains_evaluation_answers"])
        self.assertEqual("transductive_retrieval", retrieval["evaluation_regime"])
        self.assertFalse(retrieval["official_heldout_eligible"])
        self.assertEqual(
            "parametric_only_when_retrieval_unsupported",
            retrieval["fallback_policy"],
        )
        self.assertFalse(retrieval["web_search_enabled"])

        manifest = json.loads(
            (
                ROOT
                / "artifacts/hotpotqa_transductive_qa_memory_v1/index/manifest.json"
            ).read_text(encoding="utf-8")
        )
        for field, expected in (
            ("source_record_count", 640),
            ("source_train_count", 512),
            ("source_evaluation_count", 128),
            ("frozen_validation_count", 128),
            ("evaluation_overlap_count", 128),
            ("contains_evaluation_answers", True),
            ("evaluation_regime", "transductive_retrieval"),
            ("official_heldout_eligible", False),
        ):
            self.assertEqual(expected, manifest[field])

    def test_prepared_tasks_match_round01_task_projection(self) -> None:
        def projection(path: Path) -> list[tuple[str, str, str]]:
            return [
                (row["task_id"], row["question"], row["ground_truth"])
                for row in (
                    json.loads(line)
                    for line in path.read_text(encoding="utf-8").splitlines()
                    if line.strip()
                )
            ]

        original = projection(
            ROOT / "artifacts/hotpotqa_round_01/selected_tasks.jsonl"
        )
        prepared = projection(
            ROOT
            / "artifacts/hotpotqa_round01_transductive_qa_memory_v19/selected_tasks.jsonl"
        )
        self.assertEqual(128, len(original))
        self.assertEqual(original, prepared)

    def test_tool_registry_identifies_transductive_memory_without_web_search(self) -> None:
        registry = build_hotpotqa_embedding_tool_registry(
            _Index(),
            task_id="hotpotqa:frozen-evaluation",
            tool_id=HOTPOTQA_QA_MEMORY_TOOL_ID,
        )
        result, receipt = asyncio.run(
            registry.ainvoke_with_receipt(
                HOTPOTQA_QA_MEMORY_TOOL_ID,
                ToolRequest("search", {"query": "Alpha", "k": 2}),
            )
        )
        self.assertIsNotNone(result)
        assert result is not None
        identity = result.value["retrieval_index"]
        self.assertEqual("transductive_qa_memory", identity["corpus_kind"])
        self.assertEqual(640, identity["source_record_count"])
        self.assertEqual(128, identity["evaluation_overlap_count"])
        self.assertTrue(identity["contains_evaluation_answers"])
        self.assertFalse(identity["official_heldout_eligible"])
        self.assertIsNone(receipt.error_type)
        self.assertEqual((HOTPOTQA_QA_MEMORY_TOOL_ID,), registry.resource_ids)
        self.assertFalse(
            any("web" in resource.casefold() for resource in registry.resource_ids)
        )

    def test_worker_action_domain_is_search_then_all_reads_then_assessment(self) -> None:
        registry = build_hotpotqa_embedding_tool_registry(
            _Index(),
            task_id="hotpotqa:frozen-evaluation",
            tool_id=HOTPOTQA_QA_MEMORY_TOOL_ID,
        )
        adapter = HotpotQAEmbeddingReactExecutionAdapter(
            gateway=_Gateway(),
            tool_registry=registry,
            max_turns=4,
            max_tool_calls=3,
        )
        actions, complete = adapter._state_conditioned_action_domain(None, [])
        self.assertEqual(frozenset({(HOTPOTQA_QA_MEMORY_TOOL_ID, "search")}), actions)
        self.assertFalse(complete)

        observations = [
            {
                "tool_id": HOTPOTQA_QA_MEMORY_TOOL_ID,
                "observation_status": "success",
                "result": {"operation": "search", "memory_ids": ["m1", "m2"]},
            }
        ]
        actions, complete = adapter._state_conditioned_action_domain(None, observations)
        self.assertEqual(frozenset({(HOTPOTQA_QA_MEMORY_TOOL_ID, "read")}), actions)
        self.assertFalse(complete)

        for memory_id in ("m1", "m2"):
            observations.append(
                {
                    "tool_id": HOTPOTQA_QA_MEMORY_TOOL_ID,
                    "observation_status": "success",
                    "result": {"operation": "read", "memory_id": memory_id},
                }
            )
        actions, complete = adapter._state_conditioned_action_domain(None, observations)
        self.assertEqual(frozenset(), actions)
        self.assertTrue(complete)

    def test_director_has_no_tool_payload_and_relation_gate_remains_required(self) -> None:
        director = self.candidate["director"]
        graph = self.candidate["agent_graph"]
        self.assertNotIn("allowed_tools", director)
        self.assertNotIn("tools", director)
        self.assertEqual("control_plane", graph["director_feedback_mode"])
        self.assertEqual(HOTPOTQA_QA_MEMORY_TOOL_ID, graph["required_evidence_tool_id"])
        self.assertTrue(graph["require_evidence_relation"])

    def test_output_prompt_allows_parametric_fallback_only_for_unsupported(self) -> None:
        provider = ProviderSpec(
            "provider",
            kind="openai-compatible",
            endpoint="https://example.invalid/v1",
            api_key_env=None,
        )
        model = ModelSpec(
            "model",
            "provider",
            model_name="test-model",
        )
        request = AgentRequest(
            request_id="run:output:single",
            run_id="run",
            graph_revision=1,
            problem="Answer the public HotpotQA task",
            agent=AgentNode("output", "model", "return the answer"),
            model=model,
            provider=provider,
            phase=ExecutionPhase.SINGLE,
            upstream=(
                UpstreamMessage(
                    "worker",
                    "output",
                    json.dumps(
                        {
                            "retrieval_sufficiency": "unsupported",
                            "selected_memory_id": None,
                        }
                    ),
                    tool_receipts=(
                        {
                            "tool_id": HOTPOTQA_QA_MEMORY_TOOL_ID,
                            "request": {"action": "read"},
                            "error_type": None,
                        },
                    ),
                ),
            ),
            is_output_agent=True,
        )
        system = build_agent_messages(request)[0]["content"]
        self.assertIn(
            "Only retrieval_sufficiency=unsupported admits parametric fallback",
            system,
        )
        self.assertIn(
            "For retrieval_sufficiency=supported, consume the worker-selected",
            system,
        )


if __name__ == "__main__":
    unittest.main()
