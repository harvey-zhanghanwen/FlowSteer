from __future__ import annotations

import asyncio
import importlib.util
import json
from pathlib import Path
import re
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _load_script(name: str):
    path = PROJECT_ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_PROFILE = _load_script("select_hotpotqa_qa_memory_profile")
_SMOKE = _load_script("smoke_hotpotqa_qa_memory_retrieval")


class _FakeModel:
    def encode(self, texts, **kwargs):
        del kwargs
        rows: list[np.ndarray] = []
        for raw in texts:
            match = re.search(r"token-(\d{4})", str(raw))
            vector = np.zeros(512, dtype=np.float32)
            vector[int(match.group(1)) if match else 0] = 1.0
            rows.append(vector)
        return np.stack(rows)


def _train_row(index: int) -> dict[str, object]:
    task_id = f"hotpotqa:train-{index:04d}"
    return {
        "schema_version": "flowsteer.agentgraph.task.v1",
        "task_id": task_id,
        "question": f"Who is associated with token-{index:04d}?",
        "ground_truth": f"Answer {index:04d}",
        "split": "train",
        "metadata": {
            "dataset_key": "hotpotqa",
            # These fields are deliberately present in the aligned source.  The
            # QA-memory safe projection must not serialize them.
            "evaluator_payload": {
                "supporting_facts": {"title": ["private"]},
            },
            "sampling": {
                "selection": "sequential",
                "selection_index": index,
                "base_task_id": task_id,
                "cycled_training_sample": False,
            },
        },
    }


def _paraphrase(index: int) -> dict[str, object]:
    return {
        "source_train_task_id": f"hotpotqa:train-{index:04d}",
        "paraphrase_question": f"Identify the person linked to token-{index:04d}.",
        "paraphrase_answer_statement": f"The answer is Answer {index:04d}.",
        "paraphrase_provenance": "offline-reviewed-fixture",
        "paraphrase_version": "fixture-v1",
        "semantic_preservation_attested": True,
    }


def _write_fixture(root: Path) -> Path:
    data = root / "data"
    artifacts = root / "artifacts"
    config_dir = root / "config"
    data.mkdir()
    artifacts.mkdir()
    config_dir.mkdir()
    train_path = data / "train.jsonl"
    train_path.write_text(
        "".join(json.dumps(_train_row(index)) + "\n" for index in range(512)),
        encoding="utf-8",
    )
    validation_path = artifacts / "validation.jsonl"
    validation_path.write_text(
        "".join(
            json.dumps({"task_id": f"hotpotqa:validation-{index:04d}"}) + "\n"
            for index in range(128)
        ),
        encoding="utf-8",
    )
    paraphrase_path = artifacts / "paraphrases.jsonl"
    paraphrase_path.write_text(
        "".join(json.dumps(_paraphrase(index)) + "\n" for index in range(512)),
        encoding="utf-8",
    )
    config_path = config_dir / "evaluation.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "qa_embedding_retrieval": {
                    "corpus_kind": "train_qa_memory",
                    "train_tasks": str(train_path),
                    "train_sample_count": 512,
                    "frozen_validation_tasks": str(validation_path),
                    "validation_sample_count": 128,
                    "paraphrase_materialization_path": str(paraphrase_path),
                    "development_sample_count": 16,
                    "embedding_model": "fake",
                    "embedding_model_id": "fake-one-hot",
                    "embedding_device": "cpu",
                    "search_top_k": 2,
                    "tool_timeout_seconds": 5.0,
                }
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return config_path


class HotpotQAQAMemoryProfileSmokeTests(unittest.TestCase):
    def test_profile_uses_train_base_task_targets_only(self) -> None:
        with TemporaryDirectory() as temporary:
            config_path = _write_fixture(Path(temporary))
            with patch.object(
                _PROFILE, "_load_sentence_transformer", return_value=_FakeModel()
            ):
                profile = _PROFILE.select_profile(config_path)
            self.assertEqual(2, profile["selected_top_k"])
            self.assertEqual(
                {"2": 1.0, "3": 1.0, "4": 1.0, "5": 1.0},
                profile["base_task_hit_rate_by_top_k"],
            )
            self.assertEqual(
                {"2": 1.0, "3": 1.0, "4": 1.0, "5": 1.0},
                profile["mean_reciprocal_rank_by_top_k"],
            )
            self.assertEqual(512, profile["train_record_count"])
            self.assertEqual(0, profile["cycled_record_count"])
            self.assertFalse(profile["validation_question_consulted"])
            self.assertFalse(profile["validation_answer_or_alias_consulted"])
            self.assertFalse(profile["validation_supporting_facts_consulted"])
            encoded = json.dumps(profile)
            self.assertNotIn("validation-0000", encoded)
            self.assertNotIn("private", encoded)

    def test_smoke_rebuilds_deterministically_and_invokes_dynamic_tool(self) -> None:
        with TemporaryDirectory() as temporary:
            config_path = _write_fixture(Path(temporary))
            with patch(
                "src.interactive.hotpotqa_qa_memory_index._load_sentence_transformer",
                return_value=_FakeModel(),
            ):
                value = asyncio.run(
                    _SMOKE.smoke(config_path)
                )
            self.assertTrue(value["passed"])
            self.assertEqual(512, value["train_record_count"])
            self.assertEqual(128, value["heldout_validation_count"])
            self.assertEqual(0, value["validation_overlap_count"])
            self.assertTrue(value["deterministic_rebuild"])
            self.assertTrue(value["same_query_top_k_deterministic"])
            self.assertTrue(value["private_validation_or_evaluator_fields_absent"])
            self.assertTrue(value["dynamic_search_read_receipt_fields_present"])
            self.assertFalse(value["web_search_used"])
            receipts = value["tool_receipts"]
            self.assertEqual(["search", "read"], [
                receipt["request"]["action"] for receipt in receipts
            ])
            self.assertTrue(all(receipt["error_type"] is None for receipt in receipts))


if __name__ == "__main__":
    unittest.main()
