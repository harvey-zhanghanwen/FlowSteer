from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest

import yaml

from src.interactive.model_registry import ModelRegistry


ROOT = Path(__file__).resolve().parents[2]


def _load_runner():
    path = ROOT / "scripts" / "evaluate_hotpotqa_round.py"
    scripts_path = str(ROOT / "scripts")
    if scripts_path not in sys.path:
        sys.path.insert(0, scripts_path)
    spec = importlib.util.spec_from_file_location(
        "evaluate_hotpotqa_round_v23_thinking", path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class HotpotQARound01FullDatasetFactMemoryV23ThinkingProfileTests(
    unittest.TestCase
):
    @classmethod
    def setUpClass(cls) -> None:
        cls.runner = _load_runner()
        cls.v22 = yaml.safe_load(
            (
                ROOT
                / "config/evaluation_hotpotqa_round01_full_dataset_fact_memory_v22.yaml"
            ).read_text(encoding="utf-8")
        )
        cls.v23 = yaml.safe_load(
            (
                ROOT
                / "config/evaluation_hotpotqa_round01_full_dataset_fact_memory_v23_thinking.yaml"
            ).read_text(encoding="utf-8")
        )

    def test_v23_is_a_distinct_future_only_condition(self) -> None:
        self.assertNotEqual(
            self.v22["experiment"]["condition_id"],
            self.v23["experiment"]["condition_id"],
        )
        self.assertNotEqual(
            self.v22["experiment"]["output_dir"],
            self.v23["experiment"]["output_dir"],
        )
        self.assertNotEqual(
            self.v22["agent_graph"]["model_catalog_path"],
            self.v23["agent_graph"]["model_catalog_path"],
        )
        self.runner.validate_hotpot_config(self.v23)

    def test_only_worker_decoding_and_versioned_receipts_change(self) -> None:
        for section in (
            "data",
            "director",
            "evaluation",
            "grpo",
            "policy_sync",
            "exploration",
            "skills",
            "gpu",
            "deployment",
        ):
            self.assertEqual(self.v22[section], self.v23[section])

        v22_graph = dict(self.v22["agent_graph"])
        v23_graph = dict(self.v23["agent_graph"])
        v22_graph.pop("model_catalog_path")
        v23_graph.pop("model_catalog_path")
        self.assertEqual(v22_graph, v23_graph)

        v22_retrieval = dict(self.v22["qa_embedding_retrieval"])
        v23_retrieval = dict(self.v23["qa_embedding_retrieval"])
        v22_retrieval.pop("condition_id")
        v23_retrieval.pop("condition_id")
        self.assertEqual(v22_retrieval, v23_retrieval)

    def test_qwen_worker_native_thinking_is_enabled(self) -> None:
        catalog = ModelRegistry.from_yaml(
            ROOT / self.v23["agent_graph"]["model_catalog_path"]
        )
        model = catalog.require_model("qwen3.5-9b-local")
        self.assertEqual(
            "true", model.metadata["chat_template_enable_thinking"]
        )
        self.assertEqual(
            "512", model.metadata["chat_template_thinking_budget"]
        )
        self.assertEqual("4608", model.metadata["max_tokens"])


if __name__ == "__main__":
    unittest.main()
