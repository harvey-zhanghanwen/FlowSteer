from __future__ import annotations

import asyncio
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np

from src.interactive.hotpotqa_qa_memory_index import (
    HotpotQAQAMemoryIndex,
    HotpotQATrainQASource,
    build_hotpotqa_qa_memory_index,
    load_hotpotqa_train_qa_sources,
    materialize_hotpotqa_qa_memories,
)


class _FakeModel:
    def __init__(self) -> None:
        self.encoded_texts: list[str] = []

    def encode(self, texts, **kwargs):
        del kwargs
        self.encoded_texts.extend(str(text) for text in texts)
        rows = []
        for text in texts:
            lowered = str(text).casefold()
            vector = np.asarray(
                [
                    lowered.count("alpha") + 0.1,
                    lowered.count("beta") + 0.1,
                    lowered.count("bridge") + 0.1,
                ],
                dtype=np.float32,
            )
            rows.append(vector / np.linalg.norm(vector))
        return np.stack(rows)


def _aligned_record(
    task_id: str,
    *,
    index: int,
    question: str,
    answer: str,
    cycled: bool = False,
    base_task_id: str | None = None,
    split: str = "train",
) -> dict[str, object]:
    base = base_task_id or task_id
    return {
        "schema_version": "flowsteer.agentgraph.task.v1",
        "task_id": task_id,
        "question": f"Private public passages here.\n\nQuestion: {question}",
        "ground_truth": answer,
        "split": split,
        "metadata": {
            "dataset_key": "hotpotqa",
            # The source adapter must project these evaluator-private fields out.
            "evaluator_payload": {
                "supporting_facts": {"title": ["Never serialize me"]}
            },
            "sampling": {
                "selection": "sequential",
                "selection_index": index,
                "base_task_id": base,
                "cycled_training_sample": cycled,
            },
        },
    }


def _paraphrase(source_id: str, question: str, answer: str) -> dict[str, object]:
    return {
        "source_train_task_id": source_id,
        "paraphrase_question": question,
        "paraphrase_answer_statement": f"The answer is {answer}.",
        "paraphrase_provenance": "offline-reviewed-fixture",
        "paraphrase_version": "fixture-v1",
        "semantic_preservation_attested": True,
    }


class HotpotQAQAMemoryIndexTests(unittest.TestCase):
    def test_train_projection_is_disjoint_and_drops_evaluator_private_fields(self) -> None:
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "train.jsonl"
            rows = [
                _aligned_record(
                    "hotpotqa:train-a", index=0, question="Who is Alpha?", answer="Ada"
                ),
                _aligned_record(
                    "hotpotqa:train-a:cycle-0001",
                    index=1,
                    question="Who is Alpha?",
                    answer="Ada",
                    cycled=True,
                    base_task_id="hotpotqa:train-a",
                ),
            ]
            path.write_text(
                "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
            )
            sources = load_hotpotqa_train_qa_sources(
                path,
                validation_task_ids=["hotpotqa:validation"],
                expected_train_count=2,
                expected_validation_count=1,
            )
            self.assertEqual(2, len(sources))
            self.assertTrue(sources[1].cycled)
            self.assertEqual("Who is Alpha?", sources[0].question)
            encoded = json.dumps([source.__dict__ if hasattr(source, "__dict__") else {
                "source_train_task_id": source.source_train_task_id,
                "base_task_id": source.base_task_id,
                "cycled": source.cycled,
                "question": source.question,
                "canonical_answer": source.canonical_answer,
            } for source in sources])
            self.assertNotIn("supporting_facts", encoded)
            self.assertNotIn("evaluator_payload", encoded)

    def test_train_projection_rejects_validation_overlap(self) -> None:
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "train.jsonl"
            path.write_text(
                json.dumps(
                    _aligned_record(
                        "hotpotqa:same", index=0, question="Question?", answer="Answer"
                    )
                )
                + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "overlap"):
                load_hotpotqa_train_qa_sources(
                    path,
                    validation_task_ids=["hotpotqa:same"],
                    expected_train_count=1,
                    expected_validation_count=1,
                )

    def test_materialization_fails_closed(self) -> None:
        source = HotpotQATrainQASource(
            source_train_task_id="hotpotqa:train",
            base_task_id="hotpotqa:train",
            cycled=False,
            question="Who is Alpha?",
            canonical_answer="Ada Lovelace",
        )
        identity = _paraphrase("hotpotqa:train", "Who is Alpha?", "Ada Lovelace")
        with self.assertRaisesRegex(ValueError, "identical"):
            materialize_hotpotqa_qa_memories([source], [identity])

        lost_span = _paraphrase(
            "hotpotqa:train", "Identify the person called Alpha.", "another person"
        )
        with self.assertRaisesRegex(ValueError, "canonical answer span"):
            materialize_hotpotqa_qa_memories([source], [lost_span])

        private = _paraphrase(
            "hotpotqa:train", "Identify the person called Alpha.", "Ada Lovelace"
        )
        private["supporting_facts"] = {"title": ["leak"]}
        with self.assertRaisesRegex(ValueError, "private field"):
            materialize_hotpotqa_qa_memories([source], [private])

        unattested = _paraphrase(
            "hotpotqa:train", "Identify the person called Alpha.", "Ada Lovelace"
        )
        unattested["semantic_preservation_attested"] = False
        with self.assertRaisesRegex(ValueError, "attestation"):
            materialize_hotpotqa_qa_memories([source], [unattested])

    def test_build_open_search_and_read_are_deterministic(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            train_path = root / "train.jsonl"
            rows = [
                _aligned_record(
                    "hotpotqa:train-a",
                    index=0,
                    question="Who crossed the alpha bridge?",
                    answer="Ada",
                ),
                _aligned_record(
                    "hotpotqa:train-b",
                    index=1,
                    question="Who crossed the beta bridge?",
                    answer="Grace",
                ),
            ]
            train_path.write_text(
                "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
            )
            paraphrases = [
                _paraphrase(
                    "hotpotqa:train-a", "Which person traversed the alpha bridge?", "Ada"
                ),
                _paraphrase(
                    "hotpotqa:train-b", "Which person traversed the beta bridge?", "Grace"
                ),
            ]
            fake_model = _FakeModel()
            index_dir = root / "index"
            with patch(
                "src.interactive.hotpotqa_qa_memory_index._load_sentence_transformer",
                return_value=fake_model,
            ):
                manifest = build_hotpotqa_qa_memory_index(
                    index_dir=index_dir,
                    train_jsonl=train_path,
                    validation_task_ids=["hotpotqa:validation"],
                    paraphrases=paraphrases,
                    embedding_model_path="fake",
                    embedding_model_id="fake",
                    embedding_device="cpu",
                    frozen_top_k=1,
                    expected_train_count=2,
                    expected_validation_count=1,
                )
                index = HotpotQAQAMemoryIndex.open(index_dir)

            self.assertEqual(2, manifest.unique_source_count)
            self.assertEqual(0, manifest.cycled_record_count)
            self.assertEqual(2, manifest.paraphrase_count)
            self.assertTrue(
                all(text.startswith("Question:") and "\nAnswer:" in text for text in fake_model.encoded_texts[:2])
            )
            self.assertFalse(any("Who crossed" in text for text in fake_model.encoded_texts))
            artifact_text = (index_dir / "memories.jsonl").read_text(encoding="utf-8")
            self.assertNotIn("supporting_facts", artifact_text)
            self.assertNotIn("evaluator_payload", artifact_text)

            first = asyncio.run(index.search("alpha bridge", 1))
            second = asyncio.run(index.search("alpha bridge", 1))
            self.assertEqual(first, second)
            self.assertEqual("hotpotqa:train-a", first[0].source_train_task_id)
            memory = index.read(first[0].memory_id)
            self.assertEqual("Ada", memory.canonical_answer)
            with self.assertRaises(KeyError):
                index.read("missing-memory")
            with self.assertRaises(ValueError):
                asyncio.run(index.search("alpha bridge", 2))


if __name__ == "__main__":
    unittest.main()
