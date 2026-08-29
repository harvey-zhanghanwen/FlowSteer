from __future__ import annotations

import asyncio
from dataclasses import dataclass
import importlib.util
from pathlib import Path
import unittest

import yaml

from src.interactive.hotpotqa_embedding_tool import (
    HOTPOTQA_QA_MEMORY_TOOL_ID,
    build_hotpotqa_embedding_tool_registry,
)
from src.interactive.hotpotqa_full_dataset_qa_memory_index import (
    FULL_DATASET_EVALUATION_SCOPE,
    FULL_DATASET_QA_MEMORY_CORPUS_VERSION,
)
from src.interactive.tool_runtime import ToolRequest


ROOT = Path(__file__).resolve().parents[2]


def _load_runner():
    path = ROOT / "scripts" / "evaluate_hotpotqa_round.py"
    spec = importlib.util.spec_from_file_location("evaluate_hotpotqa_round_v20", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_RUNNER = _load_runner()


@dataclass(frozen=True)
class _Manifest:
    schema_version: str = "flowsteer.hotpotqa.full_dataset_qa_memory_index.v1"
    index_id: str = "hotpotqa-full-dataset-test-v1"
    corpus_version: str = FULL_DATASET_QA_MEMORY_CORPUS_VERSION
    source: str = "HotpotQA native train + validation Q-A"
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
    paraphrase_count: int = 97_852
    frozen_evaluation_count: int = 128
    evaluation_overlap_count: int = 128
    contains_evaluation_answers: bool = True
    evaluation_scope: str = FULL_DATASET_EVALUATION_SCOPE
    official_heldout_eligible: bool = False
    paraphrase_versions: tuple[str, ...] = ("full-dataset-v1",)
    paraphrase_provenances: tuple[str, ...] = ("local-qwen3.5-9b",)


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
        del query
        assert k == 3
        return tuple(
            _Hit(
                f"m{rank}",
                f"hotpotqa:source-{rank}",
                f"Question {rank}?",
                f"The answer is A{rank}.",
                1.0 - rank / 10.0,
                rank,
            )
            for rank in range(1, 4)
        )

    def read(self, memory_id: str) -> _Memory:
        return _Memory(
            memory_id,
            "hotpotqa:source-1",
            "hotpotqa:source-1",
            False,
            "Question one?",
            "The answer is A1.",
            "A1",
            "full-dataset-v1",
            "local-qwen3.5-9b",
        )


class HotpotQARound01FullDatasetQAMemoryV20ProfileTests(unittest.TestCase):
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
                / "config/evaluation_hotpotqa_round01_full_dataset_qa_memory_v20.yaml"
            ).read_text(encoding="utf-8")
        )

    def test_round01_architecture_is_preserved_and_only_data_plane_changes(self) -> None:
        self.assertEqual(self.v19["director"], self.candidate["director"])
        self.assertEqual(self.v19["agent_graph"], self.candidate["agent_graph"])
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

    def test_full_dataset_scope_counts_and_worker_boundary_are_explicit(self) -> None:
        retrieval = self.candidate["qa_embedding_retrieval"]
        graph = self.candidate["agent_graph"]
        self.assertEqual("full_dataset_qa_memory", retrieval["corpus_kind"])
        self.assertEqual(97_852, retrieval["source_record_count"])
        self.assertEqual(90_447, retrieval["source_train_count"])
        self.assertEqual(7_405, retrieval["source_validation_count"])
        self.assertEqual(128, retrieval["evaluation_overlap_count"])
        self.assertEqual(
            FULL_DATASET_EVALUATION_SCOPE,
            retrieval["evaluation_scope"],
        )
        self.assertTrue(retrieval["contains_evaluation_answers"])
        self.assertFalse(retrieval["official_heldout_eligible"])
        self.assertFalse(retrieval["web_search_enabled"])
        self.assertNotIn("allowed_tools", self.candidate["director"])
        self.assertEqual("control_plane", graph["director_feedback_mode"])
        self.assertEqual(HOTPOTQA_QA_MEMORY_TOOL_ID, graph["required_evidence_tool_id"])
        self.assertTrue(graph["require_evidence_relation"])
        self.assertEqual(
            "public_task_dynamic_full_dataset_qa_memory_search_read",
            _RUNNER._input_context(self.candidate),
        )

    def test_registry_treats_full_dataset_index_as_global_qa_memory(self) -> None:
        registry = build_hotpotqa_embedding_tool_registry(
            _Index(),
            task_id="hotpotqa:fixed-evaluation",
            tool_id=HOTPOTQA_QA_MEMORY_TOOL_ID,
        )
        result, receipt = asyncio.run(
            registry.ainvoke_with_receipt(
                HOTPOTQA_QA_MEMORY_TOOL_ID,
                ToolRequest("search", {"query": "Question one", "k": 3}),
            )
        )
        self.assertIsNone(receipt.error_type)
        assert result is not None
        identity = result.value["retrieval_index"]
        self.assertEqual("full_dataset_qa_memory", identity["corpus_kind"])
        self.assertEqual(97_852, identity["source_record_count"])
        self.assertEqual(7_405, identity["source_validation_count"])
        self.assertEqual(128, identity["evaluation_overlap_count"])
        self.assertEqual(FULL_DATASET_EVALUATION_SCOPE, identity["evaluation_scope"])


if __name__ == "__main__":
    unittest.main()
