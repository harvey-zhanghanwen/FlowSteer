from __future__ import annotations

import asyncio
import importlib.util
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
import yaml

from src.interactive.hotpotqa_qa_memory_index import (
    materialize_hotpotqa_qa_memories,
)
from src.interactive.hotpotqa_transductive_qa_memory_index import (
    HotpotQATransductiveQAMemoryIndex,
    build_hotpotqa_transductive_qa_memory_index,
    load_hotpotqa_transductive_qa_sources,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _load_script(name: str):
    path = PROJECT_ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_MATERIALIZER = _load_script("materialize_hotpotqa_transductive_qa_memory")


class _FakeModel:
    def encode(self, texts, **kwargs):
        del kwargs
        rows = []
        for text in texts:
            lowered = str(text).casefold()
            vector = np.asarray(
                [
                    lowered.count("alpha") + 0.1,
                    lowered.count("beta") + 0.1,
                    lowered.count("validation") + 0.1,
                ],
                dtype=np.float32,
            )
            rows.append(vector / np.linalg.norm(vector))
        return np.stack(rows)


def _aligned_record(
    task_id: str,
    *,
    index: int,
    split: str,
    question: str,
    answer: str,
) -> dict[str, object]:
    return {
        "schema_version": "flowsteer.agentgraph.task.v1",
        "task_id": task_id,
        "question": f"Public passages.\n\nQuestion: {question}",
        "ground_truth": answer,
        "split": split,
        "metadata": {
            "dataset_key": "hotpotqa",
            "evaluator_payload": {
                "supporting_facts": {"title": ["not serialized"]},
            },
            "sampling": {
                "selection": "sequential",
                "selection_index": index,
                "base_task_id": task_id,
                "cycled_training_sample": False,
            },
        },
    }


def _paraphrase(source_id: str, question: str, answer: str) -> dict[str, object]:
    return {
        "source_train_task_id": source_id,
        "paraphrase_question": question,
        "paraphrase_answer_statement": f"The answer is {answer}.",
        "paraphrase_provenance": "offline-reviewed-transductive-fixture",
        "paraphrase_version": "transductive-fixture-v1",
        "semantic_preservation_attested": True,
    }


class HotpotQATransductiveQAMemoryIndexTests(unittest.TestCase):
    def _fixture(self, root: Path) -> tuple[Path, Path, list[dict[str, object]]]:
        train_path = root / "train.jsonl"
        evaluation_path = root / "validation.jsonl"
        train_rows = [
            _aligned_record(
                "hotpotqa:train-alpha",
                index=0,
                split="train",
                question="Who crossed the alpha bridge?",
                answer="Ada",
            ),
            _aligned_record(
                "hotpotqa:train-beta",
                index=1,
                split="train",
                question="Who crossed the beta bridge?",
                answer="Grace",
            ),
        ]
        evaluation_rows = [
            _aligned_record(
                "hotpotqa:evaluation",
                index=0,
                split="validation",
                question="Who crossed the validation bridge?",
                answer="Katherine",
            )
        ]
        train_path.write_text(
            "".join(json.dumps(row) + "\n" for row in train_rows),
            encoding="utf-8",
        )
        evaluation_path.write_text(
            "".join(json.dumps(row) + "\n" for row in evaluation_rows),
            encoding="utf-8",
        )
        paraphrases = [
            _paraphrase(
                "hotpotqa:train-alpha",
                "Which person traversed the alpha bridge?",
                "Ada",
            ),
            _paraphrase(
                "hotpotqa:train-beta",
                "Which person traversed the beta bridge?",
                "Grace",
            ),
            _paraphrase(
                "hotpotqa:evaluation",
                "Which person traversed the validation bridge?",
                "Katherine",
            ),
        ]
        return train_path, evaluation_path, paraphrases

    def test_source_projection_combines_train_then_evaluation(self) -> None:
        with TemporaryDirectory() as temporary:
            train_path, evaluation_path, paraphrases = self._fixture(Path(temporary))
            sources = load_hotpotqa_transductive_qa_sources(
                train_jsonl=train_path,
                evaluation_jsonl=evaluation_path,
                expected_train_count=2,
                expected_evaluation_count=1,
            )
            self.assertEqual(2, len(sources.train))
            self.assertEqual(1, len(sources.evaluation))
            self.assertEqual(
                [
                    "hotpotqa:train-alpha",
                    "hotpotqa:train-beta",
                    "hotpotqa:evaluation",
                ],
                [source.source_train_task_id for source in sources.combined],
            )
            memories = materialize_hotpotqa_qa_memories(
                sources.combined, paraphrases
            )
            self.assertEqual("Katherine", memories[-1].canonical_answer)
            encoded = json.dumps([memory.to_value() for memory in memories])
            self.assertNotIn("supporting_facts", encoded)
            self.assertNotIn("evaluator_payload", encoded)

    def test_build_open_and_search_reuse_existing_runtime(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            train_path, evaluation_path, paraphrases = self._fixture(root)
            index_dir = root / "index"
            with patch(
                "src.interactive.hotpotqa_transductive_qa_memory_index._load_sentence_transformer",
                return_value=_FakeModel(),
            ):
                manifest = build_hotpotqa_transductive_qa_memory_index(
                    index_dir=index_dir,
                    train_jsonl=train_path,
                    evaluation_jsonl=evaluation_path,
                    paraphrases=paraphrases,
                    embedding_model_path="fake",
                    embedding_model_id="fake",
                    embedding_device="cpu",
                    frozen_top_k=1,
                    expected_train_count=2,
                    expected_evaluation_count=1,
                )
                index = HotpotQATransductiveQAMemoryIndex.open(index_dir)

            self.assertEqual(3, manifest.source_record_count)
            self.assertEqual(2, manifest.source_train_count)
            self.assertEqual(1, manifest.source_evaluation_count)
            self.assertEqual(1, manifest.evaluation_overlap_count)
            self.assertTrue(manifest.contains_evaluation_answers)
            self.assertEqual("transductive_retrieval", manifest.evaluation_regime)
            self.assertFalse(manifest.official_heldout_eligible)
            persisted = json.loads(
                (index_dir / "manifest.json").read_text(encoding="utf-8")
            )
            self.assertTrue(persisted["contains_evaluation_answers"])
            self.assertFalse(persisted["official_heldout_eligible"])
            self.assertEqual(1, persisted["evaluation_overlap_count"])
            memories = (index_dir / "memories.jsonl").read_text(encoding="utf-8")
            self.assertIn("Katherine", memories)
            self.assertNotIn("supporting_facts", memories)
            hits = asyncio.run(index.search("validation bridge", 1))
            self.assertEqual("hotpotqa:evaluation", hits[0].source_train_task_id)
            self.assertEqual(
                "Katherine", index.read(hits[0].memory_id).canonical_answer
            )

    def test_production_config_declares_640_record_transductive_scope(self) -> None:
        config = yaml.safe_load(
            (PROJECT_ROOT / "config" / "hotpotqa_transductive_qa_memory_v1.yaml")
            .read_text(encoding="utf-8")
        )["qa_embedding_retrieval"]
        self.assertEqual("transductive_qa_memory", config["corpus_kind"])
        self.assertEqual(512, config["train_sample_count"])
        self.assertEqual(128, config["validation_sample_count"])
        self.assertEqual(640, config["source_record_count"])
        self.assertEqual(128, config["evaluation_overlap_count"])
        self.assertTrue(config["contains_evaluation_answers"])
        self.assertEqual("transductive_retrieval", config["evaluation_regime"])
        self.assertFalse(config["official_heldout_eligible"])
        self.assertNotEqual(
            "artifacts/hotpotqa_qa_memory_source_freeze_v2/index",
            config["index_dir"],
        )

    def test_materializer_marks_candidate_as_transductive(self) -> None:
        with TemporaryDirectory() as temporary:
            train_path, evaluation_path, _ = self._fixture(Path(temporary))
            sources = load_hotpotqa_transductive_qa_sources(
                train_jsonl=train_path,
                evaluation_jsonl=evaluation_path,
                expected_train_count=2,
                expected_evaluation_count=1,
            )
            candidate = _MATERIALIZER._candidate(
                sources.evaluation[0],
                {
                    "paraphrase_question": (
                        "Which person traversed the validation bridge?"
                    ),
                    "paraphrase_answer_statement": "It was Katherine.",
                },
            )
            self.assertEqual(
                "hotpotqa-transductive-qa-paraphrase-v2",
                candidate["paraphrase_version"],
            )
            self.assertIn("Katherine", candidate["paraphrase_answer_statement"])
            materialize_hotpotqa_qa_memories(
                (sources.evaluation[0],), (candidate,)
            )

    def test_train_only_memories_bootstrap_transductive_materialization(self) -> None:
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "memories.jsonl"
            path.write_text(
                json.dumps(
                    {
                        "source_train_task_id": "hotpotqa:source",
                        "paraphrase_question": "A reworded question?",
                        "paraphrase_answer_statement": "The answer is value.",
                        "paraphrase_provenance": "existing-train-materialization",
                        "paraphrase_version": "train-v1",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            self.assertEqual(
                [
                    {
                        "source_train_task_id": "hotpotqa:source",
                        "paraphrase_question": "A reworded question?",
                        "paraphrase_answer_statement": "The answer is value.",
                        "paraphrase_provenance": "existing-train-materialization",
                        "paraphrase_version": "train-v1",
                        "semantic_preservation_attested": True,
                    }
                ],
                _MATERIALIZER._bootstrap_candidates(path),
            )


if __name__ == "__main__":
    unittest.main()
