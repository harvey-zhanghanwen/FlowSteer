from __future__ import annotations

import asyncio
import importlib.util
import json
from pathlib import Path
import re
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _load_script(name: str):
    path = PROJECT_ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_SELECTOR = _load_script("select_hotpotqa_full_dataset_fact_memory_top_k")


def _task_row(
    index: int,
    *,
    split: str,
    task_id: str | None = None,
) -> dict[str, object]:
    identity = task_id or f"hotpotqa:{split}-{index:04d}"
    return {
        "schema_version": "flowsteer.agentgraph.task.v1",
        "task_id": identity,
        "question": (
            "Based on the following passages, answer the question.\n\n"
            f"[[Private Passage]] private-passage-{split}-{index}\n\n"
            f"Question: Development question {index}?"
        ),
        "ground_truth": f"private-{split}-answer-{index}",
        "split": split,
        "metadata": {
            "dataset_key": "hotpotqa",
            "sampling": {
                "selection": "sequential",
                "selection_index": index,
                "base_task_id": identity,
                "cycled_training_sample": False,
            },
            "evaluator_payload": {
                "supporting_facts": {
                    "title": [f"private-{split}-title-{index}"],
                    "sent_id": [0],
                }
            },
        },
    }


def _write_jsonl(path: Path, values: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(value) + "\n" for value in values),
        encoding="utf-8",
    )


class _FakeIndex:
    def __init__(self, memory_to_source: dict[str, str]) -> None:
        self.memory_to_source = memory_to_source
        self.queries: list[tuple[str, int]] = []
        self.manifest = SimpleNamespace(
            frozen_top_k=5,
            indexed_text_field="fact_text",
            contains_raw_questions=False,
            contains_raw_answers=False,
            evaluation_scope="in_database_transductive",
            official_heldout_eligible=False,
            facts_path="facts.jsonl",
            fact_count=len(memory_to_source),
            index_id="fixture-index-topk5",
            embedding_model="fixture-bge",
            embedding_dimension=8,
            normalized=True,
            similarity="cosine",
            document_format="declarative_fact_only",
        )

    async def search(self, query: str, k: int):
        self.queries.append((query, k))
        match = re.fullmatch(r"Rewritten development query (\d+)\?", query)
        assert match is not None
        index = int(match.group(1))
        relevant_memory_id = f"hotpotqa-fact-{index:06d}"
        relevant_rank = (1, 2, 3, 5)[index % 4]
        hits = []
        filler_index = 64
        for rank in range(1, 6):
            if rank == relevant_rank:
                memory_id = relevant_memory_id
            else:
                memory_id = f"hotpotqa-fact-{filler_index:06d}"
                filler_index += 1
            hits.append(SimpleNamespace(memory_id=memory_id, rank=rank))
        return tuple(hits)


class HotpotQAFullDatasetFactMemoryTopKSelectorTests(unittest.TestCase):
    def _fixture(self, root: Path):
        development_path = root / "train.jsonl"
        validation_path = root / "validation.jsonl"
        provenance_path = root / "provenance.jsonl"
        index_dir = root / "index"
        index_dir.mkdir()
        output_path = root / "profile_selection.json"
        # The 65th train row is deliberately present.  It must not be read or
        # queried because architecture development is frozen to the first 64.
        _write_jsonl(
            development_path,
            [_task_row(index, split="train") for index in range(65)],
        )
        _write_jsonl(
            validation_path,
            [_task_row(index, split="validation") for index in range(128)],
        )
        memory_to_source = {
            f"hotpotqa-fact-{index:06d}": f"hotpotqa:train-{index:04d}"
            for index in range(64)
        }
        memory_to_source.update(
            {
                f"hotpotqa-fact-{index + 64:06d}":
                f"hotpotqa:validation-{index:04d}"
                for index in range(128)
            }
        )
        rewritten_query_by_source = {
            f"hotpotqa:train-{index:04d}": (
                f"Rewritten development query {index}?"
            )
            for index in range(64)
        }
        rewritten_query_by_source.update(
            {
                f"hotpotqa:validation-{index:04d}": (
                    f"Unused validation rewrite {index}?"
                )
                for index in range(128)
            }
        )
        fake_index = _FakeIndex(memory_to_source)
        return (
            development_path,
            validation_path,
            provenance_path,
            index_dir,
            output_path,
            memory_to_source,
            rewritten_query_by_source,
            fake_index,
        )

    def test_uses_only_fixed_64_development_questions_and_external_join(self) -> None:
        with TemporaryDirectory() as temporary:
            (
                development_path,
                validation_path,
                provenance_path,
                index_dir,
                output_path,
                memory_to_source,
                rewritten_query_by_source,
                fake_index,
            ) = self._fixture(Path(temporary))
            with (
                patch.object(
                    _SELECTOR,
                    "_load_provenance_join",
                    return_value=(memory_to_source, rewritten_query_by_source),
                ),
                patch.object(
                    _SELECTOR,
                    "_load_index_memory_ids",
                    return_value=frozenset(memory_to_source),
                ),
                patch.object(
                    _SELECTOR.HotpotQAFullDatasetFactMemoryIndex,
                    "open",
                    return_value=fake_index,
                ),
            ):
                receipt = asyncio.run(
                    _SELECTOR.select_hotpotqa_full_dataset_fact_memory_top_k(
                        index_dir=index_dir,
                        development_tasks_path=development_path,
                        validation_tasks_path=validation_path,
                        provenance_path=provenance_path,
                        output_path=output_path,
                    )
                )

            self.assertEqual([1, 2, 3, 5], receipt["candidate_top_k"])
            self.assertEqual(5, receipt["selected_top_k"])
            self.assertEqual(64, receipt["development_task_count"])
            self.assertEqual(64, receipt["unique_development_base_task_count"])
            self.assertEqual(128, receipt["validation_identity_count"])
            self.assertEqual(0, receipt["development_validation_base_task_id_overlap_count"])
            self.assertFalse(receipt["validation_used_for_selection"])
            self.assertFalse(receipt["validation_content_read"])
            self.assertFalse(receipt["validation_question_consulted"])
            self.assertFalse(receipt["validation_answer_or_alias_consulted"])
            self.assertFalse(receipt["validation_supporting_facts_consulted"])
            self.assertFalse(receipt["validation_evaluator_payload_consulted"])
            self.assertEqual(
                [16, 32, 48, 64],
                [value["hit_count"] for value in receipt["candidate_results"]],
            )
            self.assertEqual(64, len(fake_index.queries))
            self.assertTrue(all(k == 5 for _, k in fake_index.queries))
            self.assertEqual(
                "Rewritten development query 0?", fake_index.queries[0][0]
            )
            self.assertEqual(
                "Rewritten development query 63?", fake_index.queries[-1][0]
            )
            self.assertEqual(0, receipt["raw_question_query_count"])
            self.assertEqual(64, receipt["semantic_query_rewrite_count"])
            encoded = json.dumps(receipt)
            self.assertNotIn("Development question 64", encoded)
            self.assertNotIn("private-validation-answer", encoded)
            self.assertNotIn("private-validation-title", encoded)
            self.assertEqual(receipt, json.loads(output_path.read_text()))

    def test_rejects_development_validation_identity_overlap_before_index(self) -> None:
        with TemporaryDirectory() as temporary:
            (
                development_path,
                validation_path,
                provenance_path,
                index_dir,
                output_path,
                _,
                _,
                _,
            ) = self._fixture(Path(temporary))
            validation_rows = [
                _task_row(index, split="validation") for index in range(128)
            ]
            validation_rows[0] = _task_row(
                0,
                split="validation",
                task_id="hotpotqa:train-0000",
            )
            _write_jsonl(validation_path, validation_rows)
            with patch.object(_SELECTOR, "_load_provenance_join") as provenance:
                with self.assertRaisesRegex(ValueError, "values overlap"):
                    asyncio.run(
                        _SELECTOR.select_hotpotqa_full_dataset_fact_memory_top_k(
                            index_dir=index_dir,
                            development_tasks_path=development_path,
                            validation_tasks_path=validation_path,
                            provenance_path=provenance_path,
                            output_path=output_path,
                        )
                    )
            provenance.assert_not_called()

    def test_provenance_join_uses_only_source_id_and_line_order(self) -> None:
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "provenance.jsonl"
            _write_jsonl(
                path,
                [
                    {
                        "source_train_task_id": "hotpotqa:source-a",
                        "paraphrase_question": "private rewritten question",
                        "fact_statement": "private declarative fact",
                    },
                    {
                        "source_train_task_id": "hotpotqa:source-b",
                        "paraphrase_question": "another private rewrite",
                        "fact_statement": "another private fact",
                    },
                ],
            )
            with patch.object(_SELECTOR, "FULL_DATASET_SOURCE_COUNT", 2):
                mapping, rewritten = _SELECTOR._load_provenance_join(path)
            self.assertEqual(
                {
                    "hotpotqa-fact-000000": "hotpotqa:source-a",
                    "hotpotqa-fact-000001": "hotpotqa:source-b",
                },
                mapping,
            )
            self.assertEqual(
                {
                    "hotpotqa:source-a": "private rewritten question",
                    "hotpotqa:source-b": "another private rewrite",
                },
                rewritten,
            )
            encoded = json.dumps(mapping)
            self.assertNotIn("private rewritten question", encoded)
            self.assertNotIn("private declarative fact", encoded)


if __name__ == "__main__":
    unittest.main()
