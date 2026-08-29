from __future__ import annotations

import asyncio
from dataclasses import dataclass
import importlib.util
from pathlib import Path
import unittest

import yaml

from src.interactive.hotpotqa_embedding_tool import (
    HOTPOTQA_FACT_MEMORY_TOOL_ID,
    build_hotpotqa_embedding_tool_registry,
)
from src.interactive.hotpotqa_full_dataset_fact_memory_index import (
    FULL_DATASET_EVALUATION_SCOPE,
    FULL_DATASET_FACT_MEMORY_CORPUS_VERSION,
)
from src.interactive.tool_runtime import ToolRequest


ROOT = Path(__file__).resolve().parents[2]


def _load_runner():
    path = ROOT / "scripts" / "evaluate_hotpotqa_round.py"
    spec = importlib.util.spec_from_file_location("evaluate_hotpotqa_round_v21", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_RUNNER = _load_runner()


@dataclass(frozen=True)
class _Manifest:
    schema_version: str = "flowsteer.hotpotqa.full_dataset_fact_memory_index.v1"
    index_id: str = "hotpotqa-full-dataset-fact-test-v1"
    corpus_version: str = FULL_DATASET_FACT_MEMORY_CORPUS_VERSION
    source: str = "HotpotQA declarative facts"
    source_splits: tuple[str, ...] = ("train", "validation")
    embedding_model: str = "test-encoder"
    embedding_dimension: int = 3
    normalized: bool = True
    similarity: str = "cosine"
    frozen_top_k: int = 3
    source_record_count: int = 97_852
    source_train_count: int = 90_447
    source_validation_count: int = 7_405
    unique_source_count: int = 97_852
    cycled_record_count: int = 0
    question_rewrite_count: int = 97_852
    fact_count: int = 97_852
    semantic_rewrite_coverage: float = 1.0
    frozen_evaluation_count: int = 128
    evaluation_overlap_count: int = 128
    contains_evaluation_source_facts: bool = True
    contains_raw_questions: bool = False
    contains_raw_answers: bool = False
    document_format: str = "declarative_fact_only"
    indexed_text_field: str = "fact_text"
    evaluation_scope: str = FULL_DATASET_EVALUATION_SCOPE
    official_heldout_eligible: bool = False
    paraphrase_versions: tuple[str, ...] = ("fact-v1",)
    paraphrase_provenances: tuple[str, ...] = ("local-qwen3.5-9b",)


@dataclass(frozen=True)
class _Hit:
    memory_id: str
    fact_snippet: str
    similarity: float
    rank: int


@dataclass(frozen=True)
class _Fact:
    memory_id: str
    fact_text: str


class _Index:
    manifest = _Manifest()

    def search(self, query: str, k: int) -> tuple[_Hit, ...]:
        del query
        assert k == 3
        return tuple(
            _Hit(f"m{rank}", f"Declarative fact {rank}.", 1.0 - rank / 10.0, rank)
            for rank in range(1, 4)
        )

    def read(self, memory_id: str) -> _Fact:
        return _Fact(memory_id, "Ada authored the work.")


class HotpotQARound01FullDatasetFactMemoryV21ProfileTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.v19 = yaml.safe_load(
            (
                ROOT
                / "config/evaluation_hotpotqa_round01_transductive_qa_memory_v19.yaml"
            ).read_text(encoding="utf-8")
        )
        cls.candidate = yaml.safe_load(
            (
                ROOT
                / "config/evaluation_hotpotqa_round01_full_dataset_fact_memory_v21.yaml"
            ).read_text(encoding="utf-8")
        )

    def test_round01_architecture_is_preserved_except_fact_tool_identity(self) -> None:
        self.assertEqual(self.v19["director"], self.candidate["director"])
        baseline_graph = dict(self.v19["agent_graph"])
        candidate_graph = dict(self.candidate["agent_graph"])
        baseline_graph.pop("required_evidence_tool_id", None)
        candidate_graph.pop("required_evidence_tool_id", None)
        self.assertEqual(baseline_graph, candidate_graph)
        for section in (
            "evaluation",
            "grpo",
            "policy_sync",
            "exploration",
            "skills",
            "gpu",
            "deployment",
        ):
            self.assertEqual(self.v19[section], self.candidate[section])
        _RUNNER.validate_hotpot_config(self.candidate)

    def test_fact_scope_and_worker_boundary_are_explicit(self) -> None:
        retrieval = self.candidate["qa_embedding_retrieval"]
        graph = self.candidate["agent_graph"]
        self.assertEqual("full_dataset_fact_memory", retrieval["corpus_kind"])
        self.assertEqual(97_852, retrieval["source_record_count"])
        self.assertEqual(90_447, retrieval["source_train_count"])
        self.assertEqual(7_405, retrieval["source_validation_count"])
        self.assertEqual(128, retrieval["evaluation_overlap_count"])
        self.assertEqual(FULL_DATASET_EVALUATION_SCOPE, retrieval["evaluation_scope"])
        self.assertTrue(retrieval["contains_evaluation_source_facts"])
        self.assertFalse(retrieval["contains_raw_questions"])
        self.assertFalse(retrieval["contains_raw_answers"])
        self.assertEqual("declarative_fact_only", retrieval["document_format"])
        self.assertEqual("fact_text", retrieval["indexed_text_field"])
        self.assertFalse(retrieval["official_heldout_eligible"])
        self.assertFalse(retrieval["web_search_enabled"])
        self.assertNotIn("allowed_tools", self.candidate["director"])
        self.assertEqual("control_plane", graph["director_feedback_mode"])
        self.assertEqual(HOTPOTQA_FACT_MEMORY_TOOL_ID, graph["required_evidence_tool_id"])
        self.assertTrue(graph["require_evidence_relation"])
        self.assertEqual(
            "public_task_dynamic_full_dataset_fact_memory_search_read",
            _RUNNER._input_context(self.candidate),
        )

    def test_registry_exposes_fact_only_search_and_read(self) -> None:
        registry = build_hotpotqa_embedding_tool_registry(
            _Index(),
            task_id="hotpotqa:fixed-evaluation",
            tool_id=HOTPOTQA_FACT_MEMORY_TOOL_ID,
        )
        search_result, search_receipt = asyncio.run(
            registry.ainvoke_with_receipt(
                HOTPOTQA_FACT_MEMORY_TOOL_ID,
                ToolRequest("search", {"query": "Who authored the work", "k": 3}),
            )
        )
        self.assertIsNone(search_receipt.error_type)
        assert search_result is not None
        identity = search_result.value["retrieval_index"]
        self.assertEqual("full_dataset_fact_memory", identity["corpus_kind"])
        self.assertFalse(identity["contains_raw_questions"])
        self.assertFalse(identity["contains_raw_answers"])
        self.assertEqual("fact_text", identity["indexed_text_field"])
        hit = search_result.value["hits"][0]
        self.assertEqual(
            {"memory_id", "fact_snippet", "similarity", "rank"}, set(hit)
        )

        read_result, read_receipt = asyncio.run(
            registry.ainvoke_with_receipt(
                HOTPOTQA_FACT_MEMORY_TOOL_ID,
                ToolRequest("read", {"memory_id": "m1"}),
            )
        )
        self.assertIsNone(read_receipt.error_type)
        assert read_result is not None
        self.assertEqual(
            {"memory_id", "fact_text"}, set(read_result.value["fact"])
        )
        forbidden = {
            "question",
            "original_question",
            "canonical_answer",
            "paraphrase_question",
            "paraphrase_answer_statement",
            "ground_truth",
        }
        self.assertTrue(forbidden.isdisjoint(read_result.value["fact"]))


if __name__ == "__main__":
    unittest.main()
