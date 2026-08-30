from __future__ import annotations

import asyncio
from dataclasses import dataclass
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import yaml

from scripts import smoke_hotpotqa_full_dataset_fact_memory_retrieval as smoke_script
from src.interactive.hotpotqa_full_dataset_fact_memory_index import (
    FULL_DATASET_EVALUATION_SCOPE,
    FULL_DATASET_FACT_MEMORY_CORPUS_VERSION,
)


@dataclass(frozen=True)
class _Manifest:
    schema_version: str = "flowsteer.hotpotqa.full_dataset_fact_memory_index.v1"
    index_id: str = "hotpotqa-full-dataset-fact-smoke-fixture-v1"
    corpus_version: str = FULL_DATASET_FACT_MEMORY_CORPUS_VERSION
    source: str = "HotpotQA declarative facts"
    source_splits: tuple[str, ...] = ("train", "validation")
    embedding_model: str = "fixture-encoder"
    embedding_dimension: int = 2
    normalized: bool = True
    similarity: str = "cosine"
    frozen_top_k: int = 2
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
    paraphrase_versions: tuple[str, ...] = ("fixture-v1",)
    paraphrase_provenances: tuple[str, ...] = ("unit-test",)

    def to_value(self) -> dict[str, object]:
        return {
            name: list(value) if isinstance(value, tuple) else value
            for name, value in self.__dict__.items()
        }


@dataclass(frozen=True)
class _RawHit:
    memory_id: str
    fact_snippet: str
    similarity: float
    rank: int
    original_question: str = "Who authored the work?"
    canonical_answer: str = "Ada Lovelace"


@dataclass(frozen=True)
class _RawFact:
    memory_id: str
    fact_text: str
    original_question: str = "Who authored the work?"
    canonical_answer: str = "Ada Lovelace"
    ground_truth: str = "Ada Lovelace"


class _Index:
    manifest = _Manifest()

    def __init__(self, *, alternate: bool = False) -> None:
        self.search_calls: list[tuple[str, int]] = []
        self.read_calls: list[str] = []
        self._alternate = alternate

    async def search(self, query: str, k: int) -> tuple[_RawHit, ...]:
        self.search_calls.append((query, k))
        hits = (
            _RawHit("fact-1", "Ada Lovelace authored the work.", 0.98, 1),
            _RawHit("fact-2", "The work was published in 1843.", 0.81, 2),
        )
        if self._alternate and len(self.search_calls) == 2:
            return tuple(reversed(hits))
        return hits

    def read(self, memory_id: str) -> _RawFact:
        self.read_calls.append(memory_id)
        facts = {
            "fact-1": "Ada Lovelace authored the work.",
            "fact-2": "The work was published in 1843.",
        }
        return _RawFact(memory_id, facts[memory_id])


def _fixture(root: Path) -> Path:
    config_dir = root / "config"
    data_dir = root / "data"
    config_dir.mkdir()
    data_dir.mkdir()
    tasks = data_dir / "development.jsonl"
    tasks.write_text(
        json.dumps(
            {
                "task_id": "hotpotqa:smoke-001",
                "question": (
                    "Based on the following passages, answer the question.\n\n"
                    "[[Public]] A work is discussed.\n\n"
                    "Question: Who authored the work?"
                ),
            }
        )
        + "\n",
        encoding="utf-8",
    )
    provenance = data_dir / "fact_provenance.jsonl"
    provenance.write_text(
        json.dumps(
            {
                "source_train_task_id": "hotpotqa:smoke-001",
                "paraphrase_question": "Identify the person who authored the work.",
                "fact_statement": "Ada Lovelace authored the work.",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    config = config_dir / "evaluation.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "qa_embedding_retrieval": {
                    "corpus_kind": "full_dataset_fact_memory",
                    "document_format": "declarative_fact_only",
                    "indexed_text_field": "fact_text",
                    "evaluation_scope": FULL_DATASET_EVALUATION_SCOPE,
                    "web_search_enabled": False,
                    "search_top_k": 2,
                    "index_dir": str(data_dir / "index"),
                    "development_tasks": str(tasks),
                    "paraphrase_materialization_path": str(provenance),
                    "embedding_model": "fixture",
                    "embedding_device": "cpu",
                    "tool_timeout_seconds": 5.0,
                    "max_turns_per_agent_call": 5,
                    "max_tool_calls_per_agent_call": 3,
                    "max_action_tokens": 512,
                }
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return config


class HotpotQAFullDatasetFactMemoryRetrievalSmokeTests(unittest.TestCase):
    def test_existing_index_is_deterministic_fact_only_and_worker_owned(self) -> None:
        with TemporaryDirectory() as temporary:
            config = _fixture(Path(temporary))
            index = _Index()
            with patch.object(
                smoke_script.HotpotQAFullDatasetFactMemoryIndex,
                "open",
                return_value=index,
            ):
                value = asyncio.run(smoke_script.smoke(config))

        self.assertTrue(value["passed"])
        self.assertTrue(value["same_query_top_k_deterministic"])
        self.assertTrue(value["fact_only_search_read_projection_valid"])
        self.assertTrue(value["qa_fields_absent_from_retrieval_payload"])
        self.assertTrue(value["qa_wire_absent_from_retrieval_payload"])
        self.assertTrue(value["worker_receipt_ownership_valid"])
        self.assertEqual(smoke_script.WORKER_AGENT_ID, value["worker_agent_id"])
        self.assertEqual(["search", "read", "read"], value["receipt_actions"])
        self.assertEqual(0, value["model_api_calls"])
        self.assertFalse(value["web_search_used"])
        self.assertEqual(
            [
                ("Identify the person who authored the work.", 2),
                ("Identify the person who authored the work.", 2),
                ("Identify the person who authored the work.", 2),
            ],
            index.search_calls,
        )
        self.assertEqual(["fact-1", "fact-2"], index.read_calls)
        serialized = json.dumps(value["tool_receipts"], ensure_ascii=False)
        for forbidden in (
            "canonical_answer",
            "ground_truth",
            "original_question",
            "paraphrase_question",
            "paraphrase_answer_statement",
        ):
            self.assertNotIn(forbidden, serialized)

    def test_changed_repeated_ranking_fails_determinism_gate(self) -> None:
        with TemporaryDirectory() as temporary:
            config = _fixture(Path(temporary))
            index = _Index(alternate=True)
            with patch.object(
                smoke_script.HotpotQAFullDatasetFactMemoryIndex,
                "open",
                return_value=index,
            ):
                value = asyncio.run(smoke_script.smoke(config))

        self.assertFalse(value["passed"])
        self.assertFalse(value["same_query_top_k_deterministic"])


if __name__ == "__main__":
    unittest.main()
