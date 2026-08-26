from __future__ import annotations

import asyncio
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import pandas as pd

from src.interactive.hotpotqa_embedding_index import (
    CORPUS_SCHEMA_VERSION,
    INDEX_SCHEMA_VERSION,
    HotpotQAEmbeddingIndex,
    HotpotQAEmbeddingIndexManifest,
    HotpotQAPassage,
    load_public_contexts,
)


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
                    lowered.count("bridge") + 0.1,
                ],
                dtype=np.float32,
            )
            rows.append(vector / np.linalg.norm(vector))
        return np.stack(rows)


def _manifest(root: Path) -> HotpotQAEmbeddingIndexManifest:
    return HotpotQAEmbeddingIndexManifest(
        schema_version=INDEX_SCHEMA_VERSION,
        index_id="test-index-v1",
        corpus_version=CORPUS_SCHEMA_VERSION,
        source="HotpotQA_HF/distractor",
        source_split="train",
        project_splits=("validation",),
        embedding_model="fake",
        embedding_model_path="fake",
        embedding_dimension=3,
        normalized=True,
        similarity="cosine",
        frozen_top_k=2,
        task_count=1,
        document_count=3,
        passage_count=3,
        passage_occurrence_count=3,
        duplicate_occurrence_count=0,
        source_files=("source.parquet",),
        passages_path="passages.jsonl",
        scopes_path="task_scopes.json",
        embeddings_path="embeddings.npy",
    )


class HotpotQAEmbeddingIndexTests(unittest.TestCase):
    def test_public_projection_does_not_materialize_private_fields(self) -> None:
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "source.parquet"
            pd.DataFrame(
                [
                    {
                        "id": "native-id",
                        "question": "private question",
                        "answer": "private answer",
                        "supporting_facts": {"title": ["Alpha"], "sent_id": [0]},
                        "context": {
                            "title": ["Alpha", "Beta"],
                            "sentences": [
                                ["Alpha public text."],
                                ["Beta public text."],
                            ],
                        },
                    }
                ]
            ).to_parquet(path)
            value = load_public_contexts([path], ["hotpotqa:native-id"])
            encoded = json.dumps(value)
            self.assertEqual(
                (("Alpha", "Alpha public text."), ("Beta", "Beta public text.")),
                value["hotpotqa:native-id"],
            )
            self.assertNotIn("private answer", encoded)
            self.assertNotIn("supporting_facts", encoded)

    def test_task_scope_and_deterministic_top_k(self) -> None:
        with TemporaryDirectory() as temporary:
            passages = (
                HotpotQAPassage("p0", "d0", "Alpha", "alpha bridge"),
                HotpotQAPassage("p1", "d1", "Beta", "beta evidence"),
                HotpotQAPassage("p2", "d2", "Other", "other text"),
            )
            model = _FakeModel()
            embeddings = model.encode(
                [f"{item.title}\n{item.text}" for item in passages]
            )
            index = HotpotQAEmbeddingIndex(
                manifest=_manifest(Path(temporary)),
                passages=passages,
                scopes={"task": ("p0", "p1")},
                embeddings=embeddings,
                model=model,
            )
            first = asyncio.run(index.search("task", "alpha", 2))
            second = asyncio.run(index.search("task", "alpha", 2))
            self.assertEqual(first, second)
            self.assertEqual("p0", first[0].passage_id)
            self.assertEqual("p0", index.read("task", "p0").passage_id)
            with self.assertRaises(ValueError):
                index.read("task", "p2")
            with self.assertRaises(KeyError):
                asyncio.run(index.search("other-task", "alpha", 2))


if __name__ == "__main__":
    unittest.main()
